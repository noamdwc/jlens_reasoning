from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


@pytest.fixture
def tokenizer():
    backend = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0,
                "[BOS]": 1,
                "[EOS]": 2,
                "[USER]": 3,
                "[ASSISTANT]": 4,
                "True": 5,
                "False": 6,
                "cat": 7,
                "true": 8,
                "TRUE": 9,
                "false": 10,
                "FALSE": 11,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[EOS]",
        additional_special_tokens=["[USER]", "[ASSISTANT]"],
        chat_template="{{ bos_token }}[USER]{{ messages[0]['content'] }}{{ eos_token }}{% if add_generation_prompt %}[ASSISTANT]{% endif %}",
    )


@pytest.fixture
def model():
    torch.manual_seed(3)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=12,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=2,
            attn_implementation="eager",
        )
    ).eval()
