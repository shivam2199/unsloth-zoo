"""
Isolated repro for unsloth issue #5230.

Bug: Fine-tuning Gemma 4 with LoRA + a label mask that keeps only the LAST
assistant turn trainable (everything else -100) produces zero gradients with
the default UNSLOTH_RETURN_LOGITS=0 path. Setting UNSLOTH_RETURN_LOGITS=1
fixes it.

This script isolates the three code paths that UNSLOTH_RETURN_LOGITS gates
(see unsloth_zoo/compiler.py:1493-1726 and unsloth_zoo/loss_utils.py:173-202)
and runs each under three label-sparsity regimes matching the reporter's
strategies (A)/(B)/(C):

    (A) dense          : no masking, all tokens trainable
    (B) assistant-turn : ~50% trainable
    (C) last-turn-only : ~11% trainable  (bug-triggering case)
    (D) pathological   : exactly one trainable token per sequence
    (E) all-masked     : zero trainable tokens (division-by-zero probe)

For each (path, regime) combination we report:
  - forward loss
  - ||d loss / d hidden_states||_1 and ||...||_inf
  - whether the gradient is numerically zero (all |g| < 1e-8)

Reference path is torch.nn.functional.cross_entropy on the dense logits.
A correctly-working kernel should give a non-zero gradient on (B), (C), (D)
and should match the reference closely. (A) is the sanity anchor. (E)
should either raise or return 0 with a warning, not silently corrupt.

Run on an H100 (or any CUDA device with >= 16GB free) from a checkout of
unsloth-zoo that has `cut_cross_entropy` installed:

    pip install unsloth_zoo[...] cut-cross-entropy
    python scripts/repro_5230_sparse_mask.py

The script is self-contained: it does NOT load a Gemma 4 checkpoint. We
synthesise (hidden_states, lm_head_weight, labels) with shapes that mirror
the reporter's setup (bsz=1, seq=700, hd=3584, vocab=262144, bf16).
"""

import os
import sys

# unsloth_zoo/__init__.py has two guards:
#   line 94:  find_spec("unsloth") must succeed
#   line 278: os.environ["UNSLOTH_IS_PRESENT"] must be set
# The second is normally set by `import unsloth` running first, but we don't
# want to pull in the full unsloth package + its heavy deps. Set it manually
# — we only need the zoo's fused-CE kernels for this repro.
os.environ["UNSLOTH_IS_PRESENT"] = "1"

# Also satisfy the find_spec guard if the real unsloth isn't pip-installed.
import importlib, importlib.util as _ilu
if _ilu.find_spec("unsloth") is None:
    _here = os.path.dirname(os.path.abspath(__file__))
    _stub_root = os.path.join(_here, "_stub")
    _stub_pkg = os.path.join(_stub_root, "unsloth")
    os.makedirs(_stub_pkg, exist_ok=True)
    with open(os.path.join(_stub_pkg, "__init__.py"), "w") as f:
        f.write("# stub: unsloth_zoo only checks find_spec('unsloth')\n")
    if _stub_root not in sys.path:
        sys.path.insert(0, _stub_root)
    importlib.invalidate_caches()

import torch
import torch.nn.functional as F


# --- Config mirroring reporter's Gemma 4 26B-A4B-it setup -------------------

BSZ = 1
SEQ_LEN = 700          # reporter's average
HIDDEN_DIM = 3584      # gemma-4 hidden size
VOCAB_SIZE = 262144    # gemma-4 vocab
DTYPE = torch.bfloat16
DEVICE = "cuda"
SEED = 3407

SPARSITY_REGIMES = [
    ("A_dense",          1.00),                   # baseline
    ("B_assistant_turns", 0.50),                  # moderate sparsity
    ("C_last_turn_only",  77 / 700),              # reporter's case ~11%
    ("D_single_token",    1 / 700),               # pathological
    ("E_all_masked",      0.0),                   # division-by-zero probe
]


# --- Helpers ----------------------------------------------------------------

def make_inputs(trainable_ratio):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    hidden_states = torch.randn(
        BSZ, SEQ_LEN, HIDDEN_DIM, dtype=DTYPE, device=DEVICE, requires_grad=True,
    )
    lm_head_weight = torch.randn(
        VOCAB_SIZE, HIDDEN_DIM, dtype=DTYPE, device=DEVICE,
    ) * 0.02
    # LoRA setup: lm_head frozen, so requires_grad_=False (matches reporter).
    lm_head_weight.requires_grad_(False)

    labels = torch.randint(0, VOCAB_SIZE, (BSZ, SEQ_LEN), device=DEVICE)
    n_trainable = max(0, int(round(SEQ_LEN * trainable_ratio)))
    mask = torch.zeros(SEQ_LEN, dtype=torch.bool, device=DEVICE)
    if n_trainable > 0:
        idx = torch.randperm(SEQ_LEN, device=DEVICE)[:n_trainable]
        mask[idx] = True
    labels[:, ~mask] = -100
    return hidden_states, lm_head_weight, labels


def grad_stats(g):
    if g is None:
        return "<None>"
    g = g.float()
    return (
        f"abs_mean={g.abs().mean().item():.3e}  "
        f"abs_max={g.abs().max().item():.3e}  "
        f"all_zero={bool((g == 0).all().item())}"
    )


def run_reference(hidden_states, lm_head_weight, labels):
    # Plain HF CE path (what UNSLOTH_RETURN_LOGITS=1 falls back to, modulo
    # scaling / softcap). Used as the ground-truth gradient.
    h = hidden_states.float()
    W = lm_head_weight.float()
    logits = F.linear(h, W)
    shift_logits = logits[:, :-1, :].contiguous().view(-1, VOCAB_SIZE)
    shift_labels = labels[:, 1:].contiguous().view(-1)
    n_items = (shift_labels != -100).sum().clamp(min=1)
    loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum") / n_items
    loss.backward()
    return loss.detach(), hidden_states.grad.detach().clone()


def run_unsloth_fused(hidden_states, lm_head_weight, labels, torch_compile=True):
    from unsloth_zoo.fused_losses import unsloth_fused_ce_loss
    n_items = (labels[..., 1:] != -100).sum().clamp(min=1)
    loss = unsloth_fused_ce_loss(
        trainer=None,
        hidden_states=hidden_states,
        lm_head_weight=lm_head_weight,
        lm_head_bias=None,
        labels=labels,
        mask=None,
        n_items=n_items,
        scaling=None,
        torch_compile=torch_compile,
    )
    loss.backward()
    return loss.detach(), hidden_states.grad.detach().clone()


def run_unsloth_fused_compiled(hidden_states, lm_head_weight, labels):
    # Exercise the Inductor-compiled chunk path (what the reporter hits in
    # production — zoo calls with torch_compile=True by default in the
    # compiler.py rewrites at cross_entropy_replacement_1:1557).
    return run_unsloth_fused(hidden_states, lm_head_weight, labels, torch_compile=True)


def run_unsloth_fused_eager(hidden_states, lm_head_weight, labels):
    # Control: same kernel without Inductor. If compiled breaks but eager
    # doesn't, the bug is in the @torch.compile wrapper, not the autograd
    # Function itself.
    return run_unsloth_fused(hidden_states, lm_head_weight, labels, torch_compile=False)


def run_cce(hidden_states, lm_head_weight, labels):
    # Path taken when NOT_RETURN_LOGITS and not requires_grad_ (LoRA default).
    # Matches the exact call at compiler.py cross_entropy_replacement_1:1527 —
    # passes unshifted hidden_states + labels. fused_linear_cross_entropy
    # forwards shift=True to cut_cross_entropy.linear_cross_entropy which
    # handles alignment internally. Do NOT pre-shift here — that would
    # double-shift and corrupt gradients.
    from unsloth_zoo.loss_utils import fused_linear_cross_entropy
    n_items = (labels[..., 1:] != -100).sum().clamp(min=1)
    loss = fused_linear_cross_entropy(
        hidden_states=hidden_states,
        lm_weight=lm_head_weight,
        labels=labels,
        num_items_in_batch=n_items,
        logit_softcapping=0,
    )
    loss.backward()
    return loss.detach(), hidden_states.grad.detach().clone()


PATHS = [
    ("reference_hf_ce", run_reference),
    ("unsloth_fused_ce_loss  compiled (prod default)", run_unsloth_fused_compiled),
    ("unsloth_fused_ce_loss  eager    (control)",     run_unsloth_fused_eager),
    ("cut_cross_entropy via fused_linear_cross_entropy", run_cce),
]


# --- Main -------------------------------------------------------------------

def main():
    assert torch.cuda.is_available(), "CUDA required"
    print(f"torch={torch.__version__}  device={torch.cuda.get_device_name(0)}")
    print(f"bsz={BSZ}  seq_len={SEQ_LEN}  hd={HIDDEN_DIM}  vocab={VOCAB_SIZE}  dtype={DTYPE}")
    print()

    # Cache reference grads per regime so we can compare.
    ref_cache = {}

    for regime_name, ratio in SPARSITY_REGIMES:
        print(f"=== regime {regime_name}  trainable_ratio={ratio:.4f} ===")
        h_ref, W_ref, y_ref = make_inputs(ratio)
        n_train = int((y_ref[..., 1:] != -100).sum().item())
        print(f"    trainable tokens (after shift): {n_train} / {BSZ * (SEQ_LEN - 1)}")

        try:
            ref_loss, ref_grad = run_reference(h_ref, W_ref, y_ref)
            ref_cache[regime_name] = (ref_loss, ref_grad)
        except Exception as e:
            print(f"    reference FAILED: {type(e).__name__}: {e}")
            continue

        for path_name, path_fn in PATHS:
            if path_fn is run_reference:
                print(f"    [{path_name:<60}] loss={ref_loss.item():.4f}  {grad_stats(ref_grad)}")
                continue
            # Fresh inputs per path (grad accumulates otherwise).
            h, W, y = make_inputs(ratio)
            try:
                loss, g = path_fn(h, W, y)
                # Cosine against reference for non-zero ref.
                ref_flat = ref_cache[regime_name][1].float().flatten()
                g_flat = g.float().flatten()
                denom = (ref_flat.norm() * g_flat.norm()).clamp(min=1e-12)
                cos = (ref_flat @ g_flat / denom).item()
                rel = (g_flat - ref_flat).norm().item() / ref_flat.norm().clamp(min=1e-12).item()
                print(
                    f"    [{path_name:<60}] loss={loss.item():.4f}  "
                    f"{grad_stats(g)}  cos_vs_ref={cos:+.4f}  rel_err={rel:.3e}"
                )
            except Exception as e:
                print(f"    [{path_name:<60}] FAILED: {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    main()
