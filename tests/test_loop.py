from clausedraft.config import load_config
from clausedraft.train.loop import _metric, resolve_output_dir


def test_resolve_output_dir_uses_the_config_and_an_override():
    config = load_config()
    assert resolve_output_dir(config, None).name == "adapter"
    override = resolve_output_dir(config, "outputs/smoke")
    assert override.name == "smoke"
    assert override.parent.name == "outputs"


def test_metric_falls_back_to_the_history():
    assert _metric({}, [{"step": 1, "loss": 1.5}, {"step": 2, "eval_loss": 1.4}], "eval_loss", "eval_loss") == 1.4
    assert _metric({"train_loss": 1.1}, [{"loss": 9.0}], "train_loss", "loss") == 1.1
