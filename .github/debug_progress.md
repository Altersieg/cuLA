# L2Norm Fusion Regression — Debug Progress

**Branch**: `debug/l2norm-fusion`
**Status**: ablation flags added; waiting for H20 benchmark data.

---

## 背景

L2norm 已成功 fuse 进 SM90 主 kernel，**正确性全部通过**（34 tests pass），
但性能出现 **2-3x regression**（所有 B/H/S 配置都退化）。

## 根因假设

Kernel 用 warp specialization 做软件流水线：
- Math0/1 (256 threads): 准备 exp(α)·Q, exp(α)·K, K@V state
- MathA (128 threads): 算 QK^T, KK^T (tensor cores)

原流水线：两者 **并行** 重叠。
Fused 之后：新增 `NormReady` cross-WG barrier → MathA 每个 block 都要等
Math0/1 算完 norm → **流水线从并行变串行**。

## 实测数据（H20，bench_no_fusion.log）

`B=1` fixed-length，no fusion vs with fusion：

| T | no fusion (ms) | with fusion (ms) | 退化 |
|---|---|---|---|
| 512 | 0.199 | 0.352 | 1.77x |
| 1024 | 0.260 | 0.691 | 2.66x |
| 4096 | 0.922 | 2.751 | **2.98x** |
| 8192 | 1.799 | 5.494 | 3.05x |
| 16384 | 3.574 | 10.903 | 3.05x |

**关键规律**：退化随 T 增大收敛到 3x。小 T 退化小（block 少）、大 T 退化稳定在 3x。
这是每 block 固定 barrier stall 的典型特征，**不需要消融实验即可确认 NormReady 是主因**。

### 3x 的数学解释

原流水线：Math0/1 和 MathA 并行，每 block ≈ α（以 Math0/1 为瓶颈）
Fused 后：串行 = (norm 计算 β) + (Math0/1 剩余 α) + (MathA α)
实测 3α → norm 计算 β ≈ α，norm 时间和一个完整 WG 的工作量相当。

## 修复方向：各 WG 独立算 norm（消除 NormReady）

**不能** Python 端预计算（等于退回 fusion 之前）。

正确做法：Math0/1 和 MathA **各自独立计算 norm**，不需要跨 WG 同步：

```
Math0/1: sQqk/sKqk → 自己算 norm → 用于 prologue (exp(α)·Q·norm)
MathA:   sQqk/sKqk → 自己算 norm → 用于 subchunk (q·norm in tensor cores)
（两者读同一份 SMEM 数据，互不干扰，并行运行）
```

代价：
- norm 算两遍，但两 WG 并行 → 额外墙钟时间 ≈ 0
- MathA 需要独立 SMEM buffer `smem_norm_aux[64][2]`（512 bytes）
- 删掉 `NormReady` barrier 和 `NormExchange`（MathA 不用了）

**预期效果**：恢复原流水线 overlap → 回到 no-fusion 性能。

## 如何 pull（H20 有本地改动冲突）

```bash
# H20 上丢弃本地改动（ablation 测试用的 flag 改动），拉最新
cd ~/cuLA
git checkout -- csrc/kda/sm90/collective/mainloop_kda_fwd.hpp
git pull
```

## 已完成

- [x] 正确性修复（K state 缺失的 norm_s）
- [x] 根因确认（NormReady barrier 打断流水线，实测 3x 符合模型）
- [x] 消融开关埋点（已推，但根据数据分析可跳过）

## 待办

- [ ] 实现修复：MathA 独立算 norm，删 NormReady（`mainloop_kda_fwd.hpp`）
  - 加 `smem_norm_aux[64][2]` 到 SharedStorage
  - `compute_aux_loop_body` 里加 norm 计算（warp reduce，不用 atomicAdd）
  - 删 NormReady barrier 及 Math0/1 端的 arrive
- [ ] 正确性测试：`pytest tests/test_kda_fused_fwd.py -k l2norm`
- [ ] bench 验证：cuLA ≈ no-fusion 性能
- [ ] 清理 ablation flags（恢复 kDoNormCompute=true 等，或整体删掉）
- [ ] PR merge（`.github/pr_body.md` 已草稿）

## 相关文件

- `csrc/kda/sm90/collective/mainloop_kda_fwd.hpp` — 主 kernel
- `csrc/kda/sm90/collective/mainloop_kda_fwd.hpp` ~L1015 — ablation flags（临时）
- `.github/pr_body.md` — PR 正文草稿
- `benchmarks/bench_kda_fused_fwd.py` — bench 脚本
- `tests/test_kda_fused_fwd.py` — 正确性测试
