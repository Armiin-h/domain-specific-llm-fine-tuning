from clausedraft.eval.metrics import Comparison, score_continuation, summarize


def test_summary_averages_base_and_adapter():
    rows = [
        Comparison(
            clause_type="Notices",
            prefix="All notices",
            gold="shall be in writing to the parties.",
            base=score_continuation("shall be in writing.", "shall be in writing to the parties."),
            adapter=score_continuation(
                "This notice shall be in writing to the parties hereof.",
                "shall be in writing to the parties.",
            ),
        )
    ]
    summary = summarize(rows)
    assert summary["prompts"] == 1
    assert summary["base_word_f1"] > 0
    assert summary["adapter_in_register"] == 1
    assert summary["adapter_nll"] is None
