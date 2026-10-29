from clausedraft.infer.health import health_report


def test_health_is_ok_only_when_a_model_is_loaded():
    ready = health_report(model_loaded=True, adapter="outputs/smoke", cuda=True)
    assert ready.ok is True
    assert ready.clause_types == 100
    assert ready.presets == ("greedy", "draft")

    missing = health_report(model_loaded=False, adapter=None, cuda=False)
    assert missing.ok is False
    assert missing.adapter is None
