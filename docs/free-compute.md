# Free and cheap compute

[`compute.md`](compute.md) says Part B is the only part that needs a rented GPU.
This page is how far the free tiers get you before that becomes true, and where
exactly they stop.

**Checked September 2026.** Free-tier limits fluctuate by design — Google does
not commit to them, and quotas are throttled dynamically based on recent usage.
Treat every number here as an estimate, not a guarantee.

## Verdict

| option | free quota | memory | verdict for Part B |
| --- | --- | --- | --- |
| **Kaggle Notebooks, 2×T4** | 30 h/wk, 9 h/session | 2 × 16 GB | **best free GPU option** |
| Colab free, T4 | ~15–30 h/wk, dynamically throttled | 16 GB | works; quota roulette |
| Colab free, TPU v5e-1 | same pool | 16 GB | *worse* than the T4 here — see below |
| Kaggle TPU v3-8 | 20 h/wk | 128 GB HBM | most free memory anywhere, JAX stack |
| [TRC](https://sites.research.google/trc/about/) | 30 days, renewable | recent grants: 64 spot v5e/v6e chips | the serious TPU option; application-based |

Short version: **use a free GPU tier to get the loops working, then rent for the
sweeps.** A 3090/4090 on vast.ai is roughly $0.20–0.40/hr, so an 18-run sweep
(6 configs × 3 seeds) costs a few dollars — less than a lost afternoon.

## Free GPU

### What fits on a 16 GB T4

Memory is not the constraint at 0.5B with LoRA. TRL lets the reference model be
the base model with adapters disabled — pass `peft_config` and leave
`ref_model=None`, and no second full copy is loaded
(`trl/trainer/dpo_trainer.py`, the `is_peft_model(model) and ref_model is None`
branch).

| phase | on a free T4 |
| --- | --- |
| B1 REINFORCE-on-LM | fine |
| B1 PPO | works at 0.5B + LoRA; policy + value head + ref + RM is tight but fits |
| B2 DPO/IPO | most comfortable — no generation inside the training loop |
| B3 GRPO | works, generation-bound; `num_generations` 4–8, short completions |
| B4 overoptimization | **breaks** — the deliverable *is* a sweep, see below |
| B5 distillation | fine (0.5B teacher, ~135M student); teacher relabeling is the slow part |
| B6 | CPU only; irrelevant |

### Three T4-specific gotchas

1. **No bf16.** T4 is Turing (sm_75); bf16 needs Ampere (sm_80+). Nearly every
   TRL and open-r1 recipe you will copy sets `bf16=True` — switch it to
   `fp16=True`. This is not cosmetic: fp16 with a gradient scaler overflows more
   readily, and in RL losses the KL term is where NaNs show up first. If a run
   goes NaN, check the logprob difference before blaming the algorithm.
2. **No FlashAttention-2** (also sm_80+). Use `attn_implementation="sdpa"`.
3. **vLLM is impractical.** It would need `dtype="half"`, and colocating it with
   training in 16 GB is a losing fight. So B3 runs on the transformers
   generation path (`use_vllm=False`, the TRL default) — the slow path that GRPO
   punishes you for. Budget accordingly before sweeping group size.

### A T4-shaped starting config

```python
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    dtype=torch.float16,           # NOT bfloat16 — T4 is sm_75
    attn_implementation="sdpa",    # NOT flash_attention_2 — needs sm_80+
)

args = GRPOConfig(
    fp16=True, bf16=False,
    num_generations=4,             # TRL default is 8; halves rollout cost
    max_completion_length=128,     # TRL default is 512
    use_vllm=False,                # the default; see gotcha 3
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
)

trainer = GRPOTrainer(
    model=model,
    args=args,
    peft_config=LoraConfig(r=16, lora_alpha=32),  # ref model comes free with this
    ...,
)
```

Note `dtype=`, not `torch_dtype=` — transformers renamed it, and TRL's
`ModelConfig` uses `dtype` at our pinned commit.

### Where free tiers actually break

Not memory — **quota**. Most of Part B is sweeps: *sweep clip range and KL
coefficient*, *sweep group size and measure advantage variance*, *plot proxy
reward vs. gold metric*. Six configs × 3 seeds is 18 runs. That is where you hit
a throttle mid-experiment, lose the session, and have nothing written down.

Two habits make this survivable, both from
[`experiment-protocol.md`](experiment-protocol.md):

- Write the phase README results row **as each run finishes**, not at the end of
  the sweep. `runs/` is gitignored and the VM is ephemeral — an unrecorded run
  is a deleted run.
- Commit before launching, so `env.json`'s `git_dirty` is `false` and the run is
  traceable to a sha that still exists.

## Free TPU

### What Colab's free TPU is now

Colab **retired TPU v2-8 from the free tier and replaced it with v5e-1**. That
matters more than it sounds: v2-8 was 8 cores × 8 GB = **64 GB aggregate**, while
v5e-1 is a **single chip with 16 GB HBM**. The free TPU now has the same memory
as the free T4 — faster (197 bf16 TFLOPS) and bf16-native, but no roomier.

### Why it is still the wrong choice for B1–B4

1. **bitsandbytes is CUDA-only.** No QLoRA, no 4-bit, no 8-bit optimizers on TPU.
   This inverts the memory comparison: 16 GB *with* 4-bit quantization holds more
   model than 16 GB without it. For this workload the v5e-1 is a downgrade from
   the T4, not an upgrade.
2. **XLA recompiles when shapes change.** PPO and GRPO generate variable-length
   completions every step — exactly the pattern that triggers recompilation
   storms. PyTorch/XLA's own docs warn against Python control flow that changes
   tensor shapes; an RL rollout loop is made of it.
3. **The ladder is TRL-anchored.** B1 says *TRL PPOTrainer*, B2 *TRL DPOTrainer*,
   B3 *cross-check against TRL's GRPOTrainer*. Going to TPU means debugging XLA
   graph breaks instead of learning what a KL coefficient does. That is a
   different curriculum, and a worthwhile one — just not this one.

### Where TPU genuinely wins

[**Tunix**](https://github.com/google/tunix) is Google's JAX-native post-training
library: SFT, preference tuning, knowledge distillation, and PPO/GRPO/GSPO, built
for TPUs. It maps nearly one-to-one onto B1/B2/B3/B5, and it is real work, not a
demo — their GRPO implementation improved Gemma 2 2B-IT by ~12% pass@1 on GSM8K.

Combined with [**TRC**](https://sites.research.google/trc/about/) — free TPU
quota for 30 days, renewable, with recent grants of 64 spot v5e or v6e chips —
that is far more compute than any free GPU tier, for an application form.
Caveats: it is application-based, participants are expected to share results
publicly (paper, blog, open source), grants are often spot/preemptible, and the
TRC FAQ notes you still pay for the small driver VM and GCS buckets. Free TPUs,
not a free project.

**A good way to use this:** treat Tunix as B6 material. B6 already asks you to
map B1–B5 onto a unifying framework and port GRPO's update into Rust. Reading
Tunix's `GRPOLearner` against TRL's `GRPOTrainer` and your own from-scratch
implementation gives you three implementations of one algorithm in two paradigms
(functional JAX vs. imperative PyTorch) — which is precisely the comprehension
check B6 is after. Apply to TRC around B5 if you want the hardware; lead time is
days.

Kaggle's **TPU v3-8 (128 GB HBM, 20 h/week)** is the exception worth remembering:
it is the most free accelerator memory available anywhere, and if you are already
in a JAX stack it beats every free GPU tier outright.

## Surviving ephemeral sessions

- Put `HF_HOME` on mounted Drive (Colab) or a Kaggle dataset. Re-downloading a
  model every session burns quota you cannot get back.
- Push checkpoints to the Hub, not to the VM disk.
- Kaggle has no persistent VM and no terminal — you commit the notebook to save
  outputs. Colab free has no background execution: closing the tab ends the run.
- Neither platform gives you the ~12 h ceiling reliably. Design runs to produce a
  usable partial result at any point, and log every step rather than only at the
  end.

---

Sources: [Colab TPU v2-8 deprecation](https://github.com/googlecolab/colabtools/issues/5566) ·
[Kaggle TPU docs](https://www.kaggle.com/docs/tpu) ·
[TPU v5e specs](https://huggingface.co/docs/optimum-tpu/en/conceptual_guides/tpu_hardware_support) ·
[Tunix](https://developers.googleblog.com/introducing-tunix-a-jax-native-library-for-llm-post-training/) ·
[TRC FAQ](https://sites.research.google/trc/faq/) ·
[PyTorch/XLA recompilation](https://docs.pytorch.org/xla/master/learn/eager.html)
