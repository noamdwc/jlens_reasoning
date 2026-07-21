"""Fixed policy constants for deterministic J-Lens sanity controls."""

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
CONTROL_CASE_KEYS = (
    "spider",
    "france_capital",
    "france_language",
    "france_continent",
    "france_currency",
)

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

CONTROL_CHECK_MAP = (
    ("identity", "identity_control"),
    ("matched_random_vector", "matched_random_vector_control"),
    ("wrong_concept", "wrong_concept_control"),
    ("random_target", "random_target_control"),
)
