import hashlib

from clausedraft.train.fingerprint import file_sha256


def test_file_sha256_matches_the_file_bytes(tmp_path):
    path = tmp_path / "train.jsonl"
    payload = b'{"messages": []}\n'
    path.write_bytes(payload)
    assert file_sha256(path) == hashlib.sha256(payload).hexdigest()
