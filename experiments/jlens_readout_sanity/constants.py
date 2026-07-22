"""Fixed policy and artifact coordinates for the J-Lens sanity experiment."""

from experiments.jlens_readout_sanity.types import ReadoutCase, SwapCase

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

READOUT_CASES = (
    ReadoutCase(
        key="spider",
        prompt="The number of legs on the animal that spins webs is",
        expected_answers=("8", "eight"),
        target_concepts=("spider",),
    ),
    ReadoutCase(
        key="france_capital",
        prompt="The capital of France is the city of",
        expected_answers=("Paris",),
        target_concepts=("France",),
        literal_argument="France",
    ),
    ReadoutCase(
        key="france_language",
        prompt="Most people in France speak",
        expected_answers=("French",),
        target_concepts=("France",),
        literal_argument="France",
    ),
    ReadoutCase(
        key="france_continent",
        prompt="France is a country on the continent of",
        expected_answers=("Europe",),
        target_concepts=("France",),
        literal_argument="France",
    ),
    ReadoutCase(
        key="france_currency",
        prompt="The single-word name for the currency now used in France is the",
        expected_answers=("Euro",),
        target_concepts=("France",),
        literal_argument="France",
    ),
)

SWAP_CASES = (
    SwapCase("spider", " spider", " ant", ("6", "six")),
    SwapCase("france_capital", " France", " China", ("Beijing",)),
    SwapCase("france_language", " France", " China", ("Chinese",)),
    SwapCase("france_continent", " France", " China", ("Asia",)),
    SwapCase("france_currency", " France", " China", ("Yuan",)),
)

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
CONTROL_CASE_KEYS = tuple(case.key for case in SWAP_CASES)

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
