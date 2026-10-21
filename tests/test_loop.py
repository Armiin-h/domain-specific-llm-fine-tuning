from clausedraft.config import load_config
from clausedraft.train.loop import resolve_output_dir


def test_resolve_output_dir_uses_the_config_and_an_override():
    config = load_config()
    assert resolve_output_dir(config, None).name == "adapter"
    override = resolve_output_dir(config, "outputs/smoke")
    assert override.name == "smoke"
    assert override.parent.name == "outputs"
