from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        act_layer: Optional[Callable[[], nn.Module]] = None,
    ) -> None:
        super().__init__()
        if act_layer is None:
            act_layer = lambda: nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = act_layer()
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = act_layer()
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act2(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(
        self,
        block: Callable[..., nn.Module],
        layers: List[int],
        num_classes: int = 10,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.in_planes = 64
        if activation == "relu":
            act_layer: Callable[[], nn.Module] = lambda: nn.ReLU(inplace=True)
            init_nonlinearity = "relu"
        elif activation == "hardswish":
            act_layer = nn.Hardswish
            init_nonlinearity = "relu"
        else:
            raise ValueError(f"unsupported activation: {activation}")

        self.conv1 = nn.Conv2d(
            3,
            self.in_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.act1 = act_layer()

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, act_layer=act_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, act_layer=act_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, act_layer=act_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, act_layer=act_layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity=init_nonlinearity)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(
        self,
        block: Callable[..., nn.Module],
        planes: int,
        blocks: int,
        stride: int,
        act_layer: Callable[[], nn.Module],
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, act_layer=act_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, act_layer=act_layer))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, activation: str = "relu") -> None:
        super().__init__()
        if activation == "relu":
            act_layer: Callable[[], nn.Module] = lambda: nn.ReLU(inplace=True)
        elif activation == "hardswish":
            act_layer = nn.Hardswish
        else:
            raise ValueError(f"unsupported activation: {activation}")
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            act_layer(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            act_layer(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            act_layer(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


@dataclass
class ModelConfig:
    name: str
    num_classes: int = 10
    activation: str = "relu"


def build_model(config: ModelConfig) -> nn.Module:
    if config.name == "resnet18":
        return ResNetCIFAR(
            BasicBlock,
            [2, 2, 2, 2],
            num_classes=config.num_classes,
            activation=config.activation,
        )
    if config.name == "small-cnn":
        return SmallCNN(num_classes=config.num_classes, activation=config.activation)
    raise ValueError(f"unsupported model: {config.name}")
