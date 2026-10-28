from clausedraft.eval.register import in_legal_register


def test_contract_wording_stays_in_register():
    text = "This Agreement shall be governed by the laws of the State of Delaware."
    assert in_legal_register(text) is True


def test_chatty_openers_are_out_of_register():
    text = "Sure, here is a clause you can paste into the contract tomorrow."
    assert in_legal_register(text) is False


def test_short_text_is_out_of_register():
    assert in_legal_register("shall agree") is False
