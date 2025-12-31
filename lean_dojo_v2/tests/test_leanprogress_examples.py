import json
from types import SimpleNamespace

from lean_dojo_v2.lean_progress import create_sample_dataset, train_steps_model


def test_create_sample_dataset_writes_expected_jsonl(tmp_path, monkeypatch):
    output_path = tmp_path / "sample.jsonl"
    monkeypatch.setattr(
        create_sample_dataset, "parse_args", lambda: SimpleNamespace(output=output_path)
    )
    create_sample_dataset.main()

    with output_path.open() as f:
        rows = [json.loads(line) for line in f]
    assert rows == create_sample_dataset.SAMPLE_DATA


def test_build_text_structure_contains_sections():
    payload = {
        "goal": "gcd n n = n",
        "prefix": "intro n",
        "tactic": "simpa",
    }
    text = train_steps_model.build_text(
        payload["goal"], payload["prefix"], payload["tactic"]
    )
    assert "Goal:\n" in text
    assert "Prefix:\n" in text
    assert "Candidate tactic:\n" in text
    assert payload["goal"] in text
    assert payload["prefix"] in text
    assert payload["tactic"] in text


class DummyTokenizer:
    def __call__(self, texts, padding, truncation, max_length):
        self.last_texts = texts
        assert padding == "max_length"
        assert truncation is True
        assert isinstance(max_length, int)
        return {"input_ids": list(range(len(texts)))}


def test_tokenize_examples_handles_missing_prefix():
    tokenizer = DummyTokenizer()
    tokenize_fn = train_steps_model.tokenize_batch(tokenizer, max_length=32)
    batch = {
        "goal": ["goal 1", "goal 2"],
        # leave out prefix to trigger default handling
        "tactic": ["tac1", "tac2"],
        "steps_remaining": [3, 0],
    }

    encoded = tokenize_fn(batch)
    assert encoded["labels"] == batch["steps_remaining"]
    assert len(tokenizer.last_texts) == 2
    assert "goal 1" in tokenizer.last_texts[0]
    assert "tac1" in tokenizer.last_texts[0]
