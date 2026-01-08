"""Registry of supported module names for grad-speedup."""

SUPPORTED_STEP_RULES = ("none", "eoss", "l0l1", "sps", "sps-momentum", "adaptive-backtracking", "sagd", "silver")
SUPPORTED_DIRECTIONS = (
    "none",
    "diag-precond",
    "gn-layerwise-exact",
    "shampoo",
    "soap",
    "sophia",
    "muon",
    "muon-curv",
)
SUPPORTED_CLIP_MODES = (
    "none",
    "global",
    "layerwise",
    "ggnc",
    "ggnc-global",
    "ggnc-layerwise",
)
SUPPORTED_SPARSITY = ("none", "linbreg")
