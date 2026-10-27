import numpy as np
import pytest

from pac_reasoning.records import load_routing_data, write_jsonl


def record(i, correct, prob=0.9, tokens=10):
    return {"id": f"q{i}", "index": i, "response": "r", "num_tokens": tokens, "latency": 1.0,
            "token_prob_mean": prob, "token_logprob_mean": np.log(prob), "correct": correct}


@pytest.fixture
def files(tmp_path):
    strong = [record(i, True, tokens=100) for i in range(4)]
    weak = [record(i, i % 2 == 0, prob=0.5 + 0.1 * i) for i in range(4)]
    paths = {k: str(tmp_path / f"{k}.jsonl") for k in ("strong", "weak", "sem", "verb")}
    write_jsonl(paths["strong"], strong[::-1])  # different order on purpose
    write_jsonl(paths["weak"], weak)
    write_jsonl(paths["sem"], [{"id": f"q{i}", "semantic_loss": 0.1 * i} for i in range(4)])
    write_jsonl(str(tmp_path / "weak.verbalized.jsonl"), [{"id": f"q{i}", "verbalized_uncertainty": 0.2} for i in range(4)])
    return paths


def test_binary_logits(files):
    d = load_routing_data(files["strong"], files["weak"], loss="binary", score="logits")
    assert d.ids == ["q0", "q1", "q2", "q3"]
    np.testing.assert_array_equal(d.losses, [0, 1, 0, 1])
    np.testing.assert_allclose(d.scores, [0.5, 0.4, 0.3, 0.2])
    np.testing.assert_array_equal(d.thinking_tokens, [100] * 4)


def test_semantic_and_verbalized(files):
    d = load_routing_data(files["strong"], files["weak"], loss="semantic", score="verbalized",
                          semantic_loss_path=files["sem"])
    np.testing.assert_allclose(d.losses, [0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(d.scores, [0.2] * 4)


def test_missing_ids_are_skipped(files, tmp_path):
    write_jsonl(files["sem"], [{"id": "q0", "semantic_loss": 0.0}, {"id": "q1", "semantic_loss": 0.5}])
    with pytest.warns(UserWarning):
        d = load_routing_data(files["strong"], files["weak"], loss="semantic", semantic_loss_path=files["sem"])
    assert d.ids == ["q0", "q1"]
