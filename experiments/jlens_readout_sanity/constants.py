"""Fixed policy and artifact coordinates for the J-Lens sanity experiment."""

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"

TOP_K = 25
WORKSPACE_LAYER_LOWER_FRACTION = 0.35
WORKSPACE_LAYER_UPPER_FRACTION = 0.80
DEFAULT_INTERVENTION_STRENGTHS = (1.0, 2.0)
DEFAULT_MINIMUM_IMPROVEMENTS = 3
DEFAULT_MAX_FORMATTING_TOKENS = 2
SPIDER_READ_MAX_RANK = 5

CONTROL_SEEDS = (
    11,
    29,
    47,
    71,
    101,
    131,
    167,
    199,
    239,
    281,
    331,
    379,
    431,
    487,
    547,
    607,
)
CONTROL_REQUIRED_CASE_COUNT = 5

CONTROL_ALPHA = 1.0
IDENTITY_ATOL = 1e-6
IDENTITY_RTOL = 1e-5
NORM_ATOL = 1e-6
NORM_RTOL = 1e-5
LOW_PRECISION_NORM_ATOL = 1e-2
LOW_PRECISION_NORM_RTOL = 1e-2
PERCENTILE_QUANTILE = 0.95
PERCENTILE_INTERPRETATION = "deterministic sanity check; not statistical significance"
WRONG_CONCEPT_REQUIRED_CASE_WINS = 4
MAX_RANDOM_VECTOR_ATTEMPTS = 1024

RANDOM_VECTOR_NAMESPACE = "jlens-control-v1"
RANDOM_TARGET_NAMESPACE = "jlens-random-target-v1"

CONTROL_CHECK_MAP = (
    ("identity", "identity_control"),
    ("matched_random_vector", "matched_random_vector_control"),
    ("wrong_concept", "wrong_concept_control"),
    ("random_target", "random_target_control"),
)
