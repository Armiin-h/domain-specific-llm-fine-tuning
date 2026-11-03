from clausedraft.api.examples import EXAMPLES
from clausedraft.infer.types import resolve_clause_type


def test_examples_use_known_clause_types_and_a_real_prefix():
    titles = [item.title for item in EXAMPLES]
    assert len(titles) == len(set(titles))
    assert len(EXAMPLES) >= 3
    for item in EXAMPLES:
        assert resolve_clause_type(item.clause_type) == item.clause_type
        assert len(item.prefix.strip()) >= 24
