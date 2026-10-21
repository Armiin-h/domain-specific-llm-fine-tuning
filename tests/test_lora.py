from clausedraft.config import load_config
from clausedraft.train.lora import peft_lora_config


def test_peft_lora_config_matches_the_yaml():
    lora = peft_lora_config(load_config().lora)
    assert lora.r == 16
    assert lora.lora_alpha == 32
    assert lora.lora_dropout == 0.05
    assert lora.bias == "none"
    assert lora.task_type == "CAUSAL_LM"
    assert "q_proj" in lora.target_modules
    assert "down_proj" in lora.target_modules
