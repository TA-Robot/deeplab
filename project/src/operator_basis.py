from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class OperatorBasisConfig:
    input_dim: int
    gamma: float = 0.1
    beta_init: float = 0.01
    operator_dropout: float = 0.0
    learnable_gamma: bool = False
    norm_type: str = "layernorm"
    norm_affine: bool = False
    seed: int | None = None
    learnable_shared: bool = False
    learnable_gates: bool = False
    rff_scales: Sequence[float] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
    poly_degrees: Sequence[int] = (2, 3, 4)
    gate_count: int = 2
    rational_alphas: Sequence[float] = (0.5, 1.0, 2.0)
    rational_eps: float = 1e-3
    diffusion_alphas: Sequence[float] = (1e-3, 3e-3, 1e-2, 3e-2)
    blur_sigmas: Sequence[float] = (0.5, 1.0, 2.0)
    lse_taus: Sequence[float] = (0.5, 1.0, 2.0, 4.0)
    neighbor_taus: Sequence[float] = (0.5, 1.0, 2.0, 4.0)
    rbf_prototypes: Sequence[int] = (16, 32)
    rbf_sigmas: Sequence[float] = (0.5, 1.0, 2.0)
    attn_dims: Sequence[int] = (8,)
    attn_taus: Sequence[float] = (0.5, 1.0)
    softsort_taus: Sequence[float] = (0.5, 1.0)
    sinkhorn_taus: Sequence[float] = (0.5, 1.0)
    sinkhorn_iters: Sequence[int] = (5,)
    num_programs: int = 8
    program_depth_range: Tuple[int, int] = (2, 4)
    program_skip_prob: float = 0.3
    program_primitives: Sequence[str] = (
        "linear",
        "sin",
        "cos",
        "gelu",
        "tanh",
        "sigmoid",
        "poly2",
        "poly3",
        "rational",
        "diffusion",
        "softpool",
        "softneighbor",
        "rbf",
        "softsort",
        "sinkhorn",
        "attention",
    )


def build_obl_config(input_dim: int, profile: str = "full", **overrides: object) -> OperatorBasisConfig:
    if profile == "mini":
        base = OperatorBasisConfig(
            input_dim=input_dim,
            rff_scales=(0.5, 1.0, 2.0, 4.0),
            poly_degrees=(2, 3),
            gate_count=2,
            rational_alphas=(),
            diffusion_alphas=(1e-3, 1e-2),
            blur_sigmas=(),
            lse_taus=(0.5, 1.0),
            neighbor_taus=(0.5, 1.0),
            rbf_prototypes=(),
            rbf_sigmas=(),
            attn_dims=(),
            attn_taus=(),
            softsort_taus=(),
            sinkhorn_taus=(),
            sinkhorn_iters=(),
            num_programs=0,
        )
    elif profile == "full":
        base = OperatorBasisConfig(input_dim=input_dim)
    else:
        raise ValueError(f"unsupported profile: {profile}")
    if overrides:
        base = replace(base, **overrides)
    return base


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        out = x * scale
        if self.weight is not None:
            out = out * self.weight
        return out


class FixedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        learnable: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        bound = 1.0 / math.sqrt(in_features)
        weight = torch.empty(out_features, in_features)
        if generator is None:
            weight.uniform_(-bound, bound)
        else:
            weight.uniform_(-bound, bound, generator=generator)
        if learnable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)

        if bias:
            bias_tensor = torch.empty(out_features)
            if generator is None:
                bias_tensor.uniform_(-bound, bound)
            else:
                bias_tensor.uniform_(-bound, bound, generator=generator)
            if learnable:
                self.bias = nn.Parameter(bias_tensor)
            else:
                self.register_buffer("bias", bias_tensor)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


@dataclass
class OperatorContext:
    x: torch.Tensor
    shared_linear: torch.Tensor
    gates: list[tuple[torch.Tensor, torch.Tensor]]
    x_pad: torch.Tensor
    x_norm: torch.Tensor


def pad_replicate_1d(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x[:, :1], x, x[:, -1:]], dim=1)


def laplacian_1d_from_pad(x_pad: torch.Tensor) -> torch.Tensor:
    left = x_pad[:, :-2]
    center = x_pad[:, 1:-1]
    right = x_pad[:, 2:]
    return left - 2.0 * center + right


def lse_pool_1d(x_pad: torch.Tensor, tau: float) -> torch.Tensor:
    left = x_pad[:, :-2]
    center = x_pad[:, 1:-1]
    right = x_pad[:, 2:]
    stacked = torch.stack([left, center, right], dim=0)
    return tau * torch.logsumexp(stacked / tau, dim=0)


def soft_neighbor_1d(x_pad: torch.Tensor, tau: float) -> torch.Tensor:
    left = x_pad[:, :-2]
    center = x_pad[:, 1:-1]
    right = x_pad[:, 2:]
    dist = torch.stack([torch.abs(left - center), torch.zeros_like(center), torch.abs(right - center)], dim=0)
    weights = torch.softmax(-dist / tau, dim=0)
    vals = torch.stack([left, center, right], dim=0)
    return (weights * vals).sum(dim=0)


def gaussian_kernel_1d(sigma: float, max_radius: int = 3) -> torch.Tensor:
    radius = max(1, int(math.ceil(max_radius * sigma)))
    size = radius * 2 + 1
    xs = torch.arange(size, dtype=torch.float32) - radius
    kernel = torch.exp(-0.5 * (xs / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def sinkhorn(log_alpha: torch.Tensor, n_iters: int) -> torch.Tensor:
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return log_alpha.exp()


class BaseOperator(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def forward(self, x: torch.Tensor, ctx: OperatorContext) -> torch.Tensor:
        raise NotImplementedError


class RFFOperator(BaseOperator):
    def __init__(self, scale: float, kind: str) -> None:
        super().__init__(f"rff_{kind}_{scale:g}")
        self.scale = scale
        self.kind = kind

    def forward(self, x: torch.Tensor, ctx: OperatorContext) -> torch.Tensor:
        z = ctx.shared_linear
        if self.kind == "sin":
            return torch.sin(self.scale * z)
        return torch.cos(self.scale * z)


class PolynomialOperator(BaseOperator):
    def __init__(self, degree: int) -> None:
        super().__init__(f"poly_{degree}")
        self.degree = degree

    def forward(self, x: torch.Tensor, ctx: OperatorContext) -> torch.Tensor:
        return x.pow(self.degree)


class RationalOperator(BaseOperator):
    def __init__(self, alpha: float, eps: float) -> None:
        super().__init__(f"rational_{alpha:g}")
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor, ctx: OperatorContext) -> torch.Tensor:
        denom = 1.0 + self.alpha * x.pow(2)
        return x / (denom + self.eps)


class GateOperator(BaseOperator):
    def __init__(self, gate_index: int) -> None:
        super().__init__(f"gate_{gate_index}")
        self.gate_index = gate_index

    def forward(self, x: torch.Tensor, ctx: OperatorContext) -> torch.Tensor:
        u, v = ctx.gates[self.gate_index]
        return u * torch.sigmoid(v)


class DiffusionOperator(BaseOperator):
    def __init__(self, alpha: float) -> None:
        super().__init__(f"diffusion_{alpha:g}")
        self.alpha = alpha

    def forward(self, x: torch.Tensor, ctx: OperatorContext) -> torch.Tensor:
        lap = laplacian_1d_from_pad(ctx.x_pad)
        return x + self.alpha * lap


class GaussianBlurOperator(BaseOperator):
    def __init__(self, sigma: float) -> None:
        super().__init__(f"blur_{sigma:g}")
        kernel = gaussian_kernel_1d(sigma).view(1, 1, -1)
        self.register_buffer("kernel", kernel)
        self.pad = kernel.shape[-1] // 2

    def forward(self, x: torch.Tensor, ctx: OperatorContext) -> torch.Tensor:
        x1 = x.unsqueeze(1)
        out = F.conv1d(x1, self.kernel, padding=self.pad)
        return out.squeeze(1)


class SoftPoolOperator(BaseOperator):
    def __init__(self, tau: float) -> None:
        super().__init__(f"softpool_{tau:g}")
        self.tau = tau

    def forward(self, x: torch.Tensor, ctx: OperatorContext) -> torch.Tensor:
        return lse_pool_1d(ctx.x_pad, self.tau)


class SoftNeighborOperator(BaseOperator):
    def __init__(self, tau: float) -> None:
        super().__init__(f"softneighbor_{tau:g}")
        self.tau = tau

    def forward(self, x: torch.Tensor, ctx: OperatorContext) -> torch.Tensor:
        return soft_neighbor_1d(ctx.x_pad, self.tau)


class RBFOperator(BaseOperator):
    def __init__(self, input_dim: int, num_prototypes: int, sigma: float, generator: torch.Generator | None) -> None:
        super().__init__(f"rbf_{num_prototypes}_{sigma:g}")
        prototypes = torch.empty(num_prototypes, input_dim)
        proj = torch.empty(num_prototypes, input_dim)
        if generator is None:
            prototypes.normal_()
            proj.normal_()
        else:
            prototypes.normal_(generator=generator)
            proj.normal_(generator=generator)
        self.register_buffer("prototypes", prototypes)
        self.register_buffer("proj", proj)
        self.sigma = sigma

    def forward(self, x: torch.Tensor, ctx: OperatorContext) -> torch.Tensor:
        diff = x.unsqueeze(1) - self.prototypes.unsqueeze(0)
        dist = diff.pow(2).sum(dim=-1)
        phi = torch.exp(-dist / (2.0 * self.sigma**2))
        return phi @ self.proj


class AttentionOperator(BaseOperator):
    def __init__(self, attn_dim: int, tau: float, generator: torch.Generator | None) -> None:
        super().__init__(f"attention_{attn_dim}_{tau:g}")
        self.attn_dim = attn_dim
        self.tau = tau
        shape = (1, attn_dim)
        self.register_buffer("w_q", self._rand_tensor(shape, generator))
        self.register_buffer("w_k", self._rand_tensor(shape, generator))
        self.register_buffer("w_v", self._rand_tensor(shape, generator))
        self.register_buffer("w_out", self._rand_tensor((attn_dim,), generator))

    def _rand_tensor(self, shape: tuple[int, ...], generator: torch.Generator | None) -> torch.Tensor:
        tensor = torch.empty(*shape)
        if generator is None:
            tensor.normal_()
        else:
            tensor.normal_(generator=generator)
        return tensor

    def forward(self, x: torch.Tensor, ctx: OperatorContext) -> torch.Tensor:
        x_tokens = x.unsqueeze(-1)
        q = x_tokens * self.w_q
        k = x_tokens * self.w_k
        v = x_tokens * self.w_v
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.attn_dim)
        weights = torch.softmax(scores / self.tau, dim=-1)
        attn = torch.matmul(weights, v)
        return (attn * self.w_out).sum(dim=-1)


class SoftSortOperator(BaseOperator):
    def __init__(self, input_dim: int, tau: float, n_iters: int) -> None:
        super().__init__(f"softsort_{tau:g}_{n_iters}")
        ref = torch.linspace(-1.0, 1.0, steps=input_dim)
        self.register_buffer("reference", ref)
        self.tau = tau
        self.n_iters = n_iters

    def forward(self, x: torch.Tensor, ctx: OperatorContext) -> torch.Tensor:
        x_norm = ctx.x_norm
        diff = torch.abs(x_norm.unsqueeze(-1) - self.reference)
        log_alpha = -diff / self.tau
        perm = sinkhorn(log_alpha, self.n_iters)
        return torch.bmm(perm, x.unsqueeze(-1)).squeeze(-1)


class SinkhornOperator(BaseOperator):
    def __init__(self, input_dim: int, tau: float, n_iters: int, generator: torch.Generator | None) -> None:
        super().__init__(f"sinkhorn_{tau:g}_{n_iters}")
        ref = torch.empty(input_dim)
        if generator is None:
            ref.uniform_(-1.0, 1.0)
        else:
            ref.uniform_(-1.0, 1.0, generator=generator)
        self.register_buffer("reference", ref)
        self.tau = tau
        self.n_iters = n_iters

    def forward(self, x: torch.Tensor, ctx: OperatorContext) -> torch.Tensor:
        x_norm = ctx.x_norm
        diff = torch.abs(x_norm.unsqueeze(-1) - self.reference)
        log_alpha = -diff / self.tau
        perm = sinkhorn(log_alpha, self.n_iters)
        return torch.bmm(perm, x.unsqueeze(-1)).squeeze(-1)


class ActivationOperator(BaseOperator):
    def __init__(self, kind: str, scale: float = 1.0) -> None:
        name = f"{kind}_{scale:g}" if kind in {"sin", "cos"} else kind
        super().__init__(f"act_{name}")
        self.kind = kind
        self.scale = scale

    def forward(self, x: torch.Tensor, ctx: OperatorContext) -> torch.Tensor:
        if self.kind == "sin":
            return torch.sin(self.scale * x)
        if self.kind == "cos":
            return torch.cos(self.scale * x)
        if self.kind == "gelu":
            return F.gelu(x)
        if self.kind == "tanh":
            return torch.tanh(x)
        if self.kind == "sigmoid":
            return torch.sigmoid(x)
        raise ValueError(f"unsupported activation: {self.kind}")


class LinearOperator(BaseOperator):
    def __init__(self, input_dim: int, generator: torch.Generator | None) -> None:
        super().__init__("linear")
        self.linear = FixedLinear(input_dim, input_dim, bias=True, learnable=False, generator=generator)

    def forward(self, x: torch.Tensor, ctx: OperatorContext) -> torch.Tensor:
        return self.linear(x)


class CompositeOperator(BaseOperator):
    def __init__(self, name: str, primitives: nn.ModuleList, use_skip: bool) -> None:
        super().__init__(name)
        self.primitives = primitives
        self.use_skip = use_skip

    def forward(self, x: torch.Tensor, ctx: OperatorContext) -> torch.Tensor:
        out = x
        for primitive in self.primitives:
            local_ctx = OperatorContext(
                x=out,
                shared_linear=out,
                gates=[],
                x_pad=pad_replicate_1d(out),
                x_norm=(out - out.mean(dim=-1, keepdim=True)) / (out.std(dim=-1, keepdim=True) + 1e-6),
            )
            out = primitive(out, local_ctx)
        if self.use_skip:
            out = out + x
        return out


class OperatorBasisLayer(nn.Module):
    def __init__(self, config: OperatorBasisConfig) -> None:
        super().__init__()
        self.input_dim = config.input_dim
        self.operator_dropout = config.operator_dropout
        self.config = config

        generator: torch.Generator | None
        if config.seed is None:
            generator = None
        else:
            generator = torch.Generator()
            generator.manual_seed(config.seed)

        self.shared_linear = FixedLinear(
            self.input_dim,
            self.input_dim,
            bias=True,
            learnable=config.learnable_shared,
            generator=generator,
        )
        self.gate_linears = nn.ModuleList(
            [
                FixedLinear(
                    self.input_dim,
                    self.input_dim * 2,
                    bias=True,
                    learnable=config.learnable_gates,
                    generator=generator,
                )
                for _ in range(config.gate_count)
            ]
        )

        operators, names = self._build_operators(config, generator)
        self.operators = nn.ModuleList(operators)
        self.operator_names = names
        self.num_operators = len(operators)

        self.norms = nn.ModuleList([self._make_norm(config) for _ in range(self.num_operators)])
        self.beta = nn.Parameter(torch.full((self.num_operators,), float(config.beta_init)))
        if config.learnable_gamma:
            self.gamma = nn.Parameter(torch.tensor(float(config.gamma)))
        else:
            self.register_buffer("gamma", torch.tensor(float(config.gamma)))

    def _make_norm(self, config: OperatorBasisConfig) -> nn.Module:
        if config.norm_type == "layernorm":
            return nn.LayerNorm(self.input_dim, elementwise_affine=config.norm_affine)
        if config.norm_type == "rmsnorm":
            return RMSNorm(self.input_dim, elementwise_affine=config.norm_affine)
        raise ValueError(f"unsupported norm_type: {config.norm_type}")

    def _build_operators(
        self, config: OperatorBasisConfig, generator: torch.Generator | None
    ) -> tuple[list[BaseOperator], list[str]]:
        ops: list[BaseOperator] = []
        names: list[str] = []

        for scale in config.rff_scales:
            op = RFFOperator(scale=scale, kind="sin")
            ops.append(op)
            names.append(op.name)
        for scale in config.rff_scales:
            op = RFFOperator(scale=scale, kind="cos")
            ops.append(op)
            names.append(op.name)

        for degree in config.poly_degrees:
            op = PolynomialOperator(degree=degree)
            ops.append(op)
            names.append(op.name)

        for gate_index in range(config.gate_count):
            op = GateOperator(gate_index=gate_index)
            ops.append(op)
            names.append(op.name)

        for alpha in config.rational_alphas:
            op = RationalOperator(alpha=alpha, eps=config.rational_eps)
            ops.append(op)
            names.append(op.name)

        for alpha in config.diffusion_alphas:
            op = DiffusionOperator(alpha=alpha)
            ops.append(op)
            names.append(op.name)

        for sigma in config.blur_sigmas:
            op = GaussianBlurOperator(sigma=sigma)
            ops.append(op)
            names.append(op.name)

        for tau in config.lse_taus:
            op = SoftPoolOperator(tau=tau)
            ops.append(op)
            names.append(op.name)

        for tau in config.neighbor_taus:
            op = SoftNeighborOperator(tau=tau)
            ops.append(op)
            names.append(op.name)

        for num_proto in config.rbf_prototypes:
            for sigma in config.rbf_sigmas:
                op = RBFOperator(self.input_dim, num_proto, sigma, generator)
                ops.append(op)
                names.append(op.name)

        for attn_dim in config.attn_dims:
            for tau in config.attn_taus:
                op = AttentionOperator(attn_dim=attn_dim, tau=tau, generator=generator)
                ops.append(op)
                names.append(op.name)

        for tau in config.softsort_taus:
            op = SoftSortOperator(self.input_dim, tau=tau, n_iters=5)
            ops.append(op)
            names.append(op.name)

        for tau in config.sinkhorn_taus:
            for n_iters in config.sinkhorn_iters:
                op = SinkhornOperator(self.input_dim, tau=tau, n_iters=n_iters, generator=generator)
                ops.append(op)
                names.append(op.name)

        if config.num_programs > 0:
            ops, names = self._build_programs(config, generator, ops, names)

        return ops, names

    def _build_programs(
        self,
        config: OperatorBasisConfig,
        generator: torch.Generator | None,
        ops: list[BaseOperator],
        names: list[str],
    ) -> tuple[list[BaseOperator], list[str]]:
        rng = random.Random(config.seed) if config.seed is not None else random
        for idx in range(config.num_programs):
            depth = rng.randint(config.program_depth_range[0], config.program_depth_range[1])
            primitives = nn.ModuleList()
            prim_names = []
            for _ in range(depth):
                prim_name = rng.choice(list(config.program_primitives))
                primitive = self._make_primitive(prim_name, config, generator, rng)
                primitives.append(primitive)
                prim_names.append(primitive.name)
            use_skip = rng.random() < config.program_skip_prob
            name = f"program_{idx}_{'-'.join(prim_names)}"
            ops.append(CompositeOperator(name=name, primitives=primitives, use_skip=use_skip))
            names.append(name)
        return ops, names

    def _make_primitive(
        self,
        prim_name: str,
        config: OperatorBasisConfig,
        generator: torch.Generator | None,
        rng: random.Random,
    ) -> BaseOperator:
        if prim_name == "linear":
            return LinearOperator(self.input_dim, generator)
        if prim_name == "sin":
            scale = rng.choice(list(config.rff_scales)) if config.rff_scales else 1.0
            return ActivationOperator(kind="sin", scale=scale)
        if prim_name == "cos":
            scale = rng.choice(list(config.rff_scales)) if config.rff_scales else 1.0
            return ActivationOperator(kind="cos", scale=scale)
        if prim_name == "gelu":
            return ActivationOperator(kind="gelu")
        if prim_name == "tanh":
            return ActivationOperator(kind="tanh")
        if prim_name == "sigmoid":
            return ActivationOperator(kind="sigmoid")
        if prim_name == "poly2":
            return PolynomialOperator(degree=2)
        if prim_name == "poly3":
            return PolynomialOperator(degree=3)
        if prim_name == "rational":
            alpha = rng.choice(list(config.rational_alphas)) if config.rational_alphas else 1.0
            return RationalOperator(alpha=alpha, eps=config.rational_eps)
        if prim_name == "diffusion":
            alpha = rng.choice(list(config.diffusion_alphas)) if config.diffusion_alphas else 1e-3
            return DiffusionOperator(alpha=alpha)
        if prim_name == "softpool":
            tau = rng.choice(list(config.lse_taus)) if config.lse_taus else 1.0
            return SoftPoolOperator(tau=tau)
        if prim_name == "softneighbor":
            tau = rng.choice(list(config.neighbor_taus)) if config.neighbor_taus else 1.0
            return SoftNeighborOperator(tau=tau)
        if prim_name == "rbf":
            num_proto = rng.choice(list(config.rbf_prototypes)) if config.rbf_prototypes else 8
            sigma = rng.choice(list(config.rbf_sigmas)) if config.rbf_sigmas else 1.0
            return RBFOperator(self.input_dim, num_proto, sigma, generator)
        if prim_name == "softsort":
            tau = rng.choice(list(config.softsort_taus)) if config.softsort_taus else 1.0
            return SoftSortOperator(self.input_dim, tau=tau, n_iters=5)
        if prim_name == "sinkhorn":
            tau = rng.choice(list(config.sinkhorn_taus)) if config.sinkhorn_taus else 1.0
            n_iters = rng.choice(list(config.sinkhorn_iters)) if config.sinkhorn_iters else 5
            return SinkhornOperator(self.input_dim, tau=tau, n_iters=n_iters, generator=generator)
        if prim_name == "attention":
            attn_dim = rng.choice(list(config.attn_dims)) if config.attn_dims else 4
            tau = rng.choice(list(config.attn_taus)) if config.attn_taus else 1.0
            return AttentionOperator(attn_dim=attn_dim, tau=tau, generator=generator)
        raise ValueError(f"unsupported primitive: {prim_name}")

    def beta_l1(self) -> torch.Tensor:
        return self.beta.abs().sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_linear = self.shared_linear(x)
        gates = []
        for gate_linear in self.gate_linears:
            u, v = gate_linear(x).chunk(2, dim=-1)
            gates.append((u, v))
        x_pad = pad_replicate_1d(x)
        x_norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
        ctx = OperatorContext(x=x, shared_linear=shared_linear, gates=gates, x_pad=x_pad, x_norm=x_norm)

        ops = [op(x, ctx) for op in self.operators]
        if len(ops) != self.num_operators:
            raise RuntimeError(f"operator count mismatch: {len(ops)} vs {self.num_operators}")

        normed_ops = [norm(op) for norm, op in zip(self.norms, ops)]
        ops_tensor = torch.stack(normed_ops, dim=0)

        if self.training and self.operator_dropout > 0:
            keep_prob = 1.0 - self.operator_dropout
            mask = (torch.rand(self.num_operators, device=x.device) < keep_prob).float()
            ops_tensor = ops_tensor * (mask[:, None, None] / keep_prob)

        mix = (ops_tensor * self.beta[:, None, None]).sum(dim=0)
        return x + self.gamma * mix
