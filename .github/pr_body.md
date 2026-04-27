<!-- .github/pull_request_template.md -->

## 📌 Description

Fuse the l2norm computation into the KDA forward kernel on SM90 Hopper, eliminating a separate l2norm kernel launch and its GMEM round-trip when `use_qk_l2norm_in_kernel=True`.

**Core changes** (`csrc/kda/sm90/collective/mainloop_kda_fwd.hpp`):

- **L2norm in Math0/1**: zero-init `smem_norm_partial[64][2]` → atomicAdd `sum(q²)` / `sum(k²)` → `rsqrt`. Synchronized via `NormExchange` (intra-WG) and `NormReady` (cross-WG with MathA).
- **Norm application — three K paths must all be patched**:
  | Path | What it feeds | Status |
  |---|---|---|
  | Math0/1 prologue (`exp(α)·Q·norm`, `exp(α)·K·norm`) | Q@KV and initial K@V WGMMA | ✅ |
  | MathA subchunk (gated `Q·norm`, `K·norm` in registers) | intra-chunk QK^T / KK^T | ✅ |
  | K state update (`exp(α_last − α)·K·norm`) | K@V WGMMA that updates KV state | ✅ **fixed in this PR** (was missing — root cause of the 22% error) |
- **Register-cached norms in MathA**: cache 20 floats/thread (4 row tiles × {Q_lo, Q_hi, K_lo, K_hi} + 4 col tiles × K^T) right after `NormReady`, avoiding repeated SMEM loads inside the head-dim loop.
- **Barrier allocation via developer API** (avoids the user-API `+8` offset that overflows SM90's 16-barrier limit):
  - `NormExchange = TmemAllocBarrier` (effective ID 6, unused on SM90)
  - `NormReady = Sm120MainloopBarrier` (effective ID 7, unused on SM90)

## 🔍 Related Issues

N/A

## 🚀 Pull Request Checklist

### ✅ Pre-commit Checks

- [ ] I have installed `pre-commit` by running `pip install pre-commit` (or used your preferred method).
- [ ] I have installed the hooks with `pre-commit install`.
- [ ] I have run the hooks manually with `pre-commit run --all-files` and fixed any reported issues.

## 🧪 Tests

- [x] Tests have been added or updated as needed. (existing `test_kda_fused_fwd.py` already exercises `use_qk_l2norm_in_kernel=True`)
- [x] All tests are passing.

```
tests/test_kda_fused_fwd.py ........................ 34 passed in 36.24s
```

## ⚡ Performance

End-to-end timing via CUDA events (`benchmarks/bench_kda_fused_fwd.py`, warmup=25, iters=100, `use_qk_l2norm_in_kernel=True` on both sides), comparing cuLA fully-fused vs FLA Triton baseline:

```
<paste the output of `python benchmarks/bench_kda_fused_fwd.py --mode both` here>
```

Expected gains: one fewer kernel launch and one fewer Q/K GMEM round-trip vs the previously non-fused cuLA path.

## Reviewer Notes

The single most consequential line in this PR is the addition of `* norm_s` in the K state-update path (`mainloop_kda_fwd.hpp` ~L1366):

```cpp
float norm_s = storage.smem_norm_partial[int(seq)][1];
auto k_scaled = Element(exp2f(alpha_last - alpha) * float(k) * norm_s);
```

K flows into the WGMMA pipeline through three independent paths (see table above); the state-update path was the only one not normalizing K, which silently broke every block after the first. Worth a careful look to confirm the SMEM index `[int(seq)][1]` matches the prologue's `[row][1]` semantics.

### Debugging history (for future reference)
1. SMEM overflow → shrink `smem_norm_partial` to `[64][2]`, share via atomicAdd
2. Barrier ID 16 > SM90 max 15 → switch to developer API (`ReservedNamedBarriers`)
3. Cross-WG sync race → split into `NormExchange` and `NormReady`
4. NaN → enable real norm values (was hardcoded `1.0f` for debug)
5. 22% numerical error → **K state-update path was missing `norm_k`** (one-line fix)
