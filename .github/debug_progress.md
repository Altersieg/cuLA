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

## 消融实验（已埋好开关）

文件: `csrc/kda/sm90/collective/mainloop_kda_fwd.hpp` ~L1015

```cpp
constexpr bool kDoNormCompute = ???;  // zero-init + atomicAdd + rsqrt
constexpr bool kDoNormBarrier = ???;  // NormReady cross-WG barrier
constexpr bool kUseRealNorm  = ???;   // apply real norm vs 1.0f
```

| 实验 | Compute | Barrier | RealNorm | 目的 |
|---|---|---|---|---|
| A | false | false | false | 纯代码膨胀/reg 压力 |
| B | false | true  | false | 纯 barrier 序列化开销 |
| C | true  | false | false | 纯 atomicAdd + rsqrt 开销 |
| Full | true | true | true | 当前完整实现 |

注意：改 flag **不影响正确性**（因为用 1.0f 代替 norm），只看性能。
正确性验证用 Full。

## 如何跑实验（H20）

```bash
# 0. 拉最新代码
cd ~/cuLA && git checkout debug/l2norm-fusion && git pull

# 1. 切换到某个实验（以 Test A 为例）
# 直接手编辑 L1015-1017 三行，或 sed 一把梭：
sed -i '0,/kDoNormCompute = [a-z]*/s//kDoNormCompute = false/' \
  csrc/kda/sm90/collective/mainloop_kda_fwd.hpp
sed -i '0,/kDoNormBarrier = [a-z]*/s//kDoNormBarrier = false/' \
  csrc/kda/sm90/collective/mainloop_kda_fwd.hpp
sed -i '0,/kUseRealNorm = [a-z]*/s//kUseRealNorm = false/' \
  csrc/kda/sm90/collective/mainloop_kda_fwd.hpp

# 2. rebuild + bench
rm -rf build
python setup.py build_ext --inplace 2>&1 | tail -5
python benchmarks/bench_kda_fused_fwd.py --mode fixed 2>&1 | tee bench_testA.log

# 3. 重复 Test B/C/Full
```

## 结果怎么看

对比 `B=1 T=4096` 那一行的 `cuLA(ms)`（最稳定的退化点）：

| 实验 | 预期 | 含义 |
|---|---|---|
| main | ~0.92 | 基线 |
| A | ≈ main? | 若是 → 代码膨胀无影响 |
| B | ≈ Full? | 若是 → **barrier 是主因** |
| C | ≈ main? | 若是 → atomicAdd/rsqrt 开销不大 |
| Full | ~2.74 | 当前 |

**最可能**: A≈main, B≈Full, C≈main → 确认是 NormReady barrier 打断流水线。

## 确认元凶后的修复方向

若 barrier 是主因，fix 选项：
1. **Python 端算 norm**（kernel 外预计算）→ 零 overhead，fusion 部分意义打折
2. **MathA 自己算 norm**（从 SMEM 读 Q/K）→ 不跨 WG，但 MathA 多轮计算
3. **Pipeline offset**（block N 的 MathA 用 block N-1 的 norm）→ 保持 overlap

## 已完成

- [x] 正确性修复（K state 缺失的 norm_s）
- [x] 消融开关埋点（推到 `debug/l2norm-fusion`）
- [x] Flags 作用域修复（移到 compute_loop_body 之外）

## 待办

- [ ] H20 跑 A/B/C/Full 四组 bench，填表
- [ ] 根据结果选择 fix 方向并实现
- [ ] 重跑 bench 验证 ≤ main 性能
- [ ] PR merge（`.github/pr_body.md` 已草稿）

## 相关文件

- `csrc/kda/sm90/collective/mainloop_kda_fwd.hpp` — 主 kernel + ablation flags
- `.github/pr_body.md` — PR 正文草稿
- `benchmarks/bench_kda_fused_fwd.py` — bench 脚本
- `tests/test_kda_fused_fwd.py` — 正确性测试
