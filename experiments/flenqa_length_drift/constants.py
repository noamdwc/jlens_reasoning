"""Fixed FLenQA readout budgets and verified task labels."""

from __future__ import annotations

PIR_TASK = "PIR"
MONOREL_TASK = "MonoRel"
RULETAKER_TASK = "Simplified RuleTaker"

MAX_SEQ_LEN = 4096
TOP_K = 25
ANCHOR_PADDING_COUNT = 4
ANCHOR_BUDGET = 12
SUMMARY_POSITION_BUDGET = 48
KEY_SPAN_SUMMARY_CAP = 12
FINAL_POSITION_COUNT = 4
SHARD_SIZE = 500
