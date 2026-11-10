"""B2: the sequence log-probabilities under right padding, and the two losses at known points."""

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "experiments" / "b2_dpo"), str(REPO / "experiments" / "b1_ppo"),
                str(REPO / "experiments" / "a4_policy_gradient"), str(REPO / "src")]
pytest.importorskip("peft")
import dpo_lm  # noqa: E402


def tiny_model():
    cfg = transformers.Qwen2Config(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64)
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(cfg).eval()


def test_seq_logprobs_ignore_padding_and_prompt():
    model = tiny_model()
    rows = [([1, 2, 3], [4, 5]), ([6, 7], [8, 9, 10, 11])]
    with torch.no_grad():
        s, n = dpo_lm.seq_logprobs(model, *dpo_lm.batch_tensors(rows, 0, "cpu"))
        for i, (p, c) in enumerate(rows):
            ids = torch.tensor([p + c])
            lp = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1)
            want = sum(lp[len(p) - 1 + j, t] for j, t in enumerate(c))
            assert n[i] == len(c)
            assert torch.allclose(s[i], want, atol=1e-4)


def test_losses_at_known_points():
    z = torch.zeros(3)
    loss, h = dpo_lm.preference_loss("dpo", 0.1, z, z, z, z)
    assert torch.allclose(loss, torch.full((3,), math.log(2))) and torch.all(h == 0)
    loss, _ = dpo_lm.preference_loss("ipo", 0.1, z, z, z, z)
    assert torch.allclose(loss, torch.full((3,), 25.0))
    # IPO's optimum: h = 1/(2τ) = 5 gives zero loss; DPO keeps falling past it.
    w = torch.full((3,), 5.0)
    assert torch.allclose(dpo_lm.preference_loss("ipo", 0.1, w, z, z, z)[0], torch.zeros(3))
    assert torch.all(dpo_lm.preference_loss("dpo", 0.1, 2 * w, z, z, z)[0]
                     < dpo_lm.preference_loss("dpo", 0.1, w, z, z, z)[0])
