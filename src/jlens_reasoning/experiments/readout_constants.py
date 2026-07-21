"""Artifact coordinates and policy constants for J-Lens readout experiments."""

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = (
    "qwen3.5-4b/jlens/Salesforce-wikitext/"
    "Qwen3.5-4B_jacobian_lens_n1000.pt"
)

TOP_K = 25
WORKSPACE_LAYER_LOWER_FRACTION = 0.35
WORKSPACE_LAYER_UPPER_FRACTION = 0.80
DEFAULT_INTERVENTION_STRENGTHS = (1.0, 2.0)
DEFAULT_MINIMUM_IMPROVEMENTS = 3
DEFAULT_MAX_FORMATTING_TOKENS = 2
SPIDER_READ_MAX_RANK = 5
