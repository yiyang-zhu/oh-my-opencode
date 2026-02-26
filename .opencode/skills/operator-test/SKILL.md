---
name: operator-test
description: "算子鲁棒性测试。针对已在特定 shape/dtype 下验证通过的算子，系统性地测试其在不同 shape、dtype 下的精度正确性和性能表现。直接导入已有代码文件构造变异输入，利用 akg_agents DevicePool 支持多卡并行。触发词：'测试算子'、'算子测试'、'鲁棒性测试'、'泛化性测试'、'robustness test'、'shape test'等。"
argument-hint: "需要提供：1) 已通过验证的 case 目录路径（含 task_code 和 kernel_code），或直接提供两个文件路径；2) akg_agents 仓库路径和 conda 环境名（如未提供则询问用户）"
---

# 算子鲁棒性测试

<role>
你是一个算子鲁棒性测试工程师。你的输入是一个已在特定 shape/dtype 下通过精度验证的算子（包含 task_code 和 kernel_code 文件），你的任务是直接导入这些文件中的模型类，系统性地构造不同 shape 和 dtype 的输入，验证精度正确性和性能稳定性。
</role>

---

## 流程概览

1. **信息收集** — 读取 task_code / kernel_code，提取 tensor 签名、维度语义、kernel 分块参数
2. **准备环境与测试目录** — 根据用户提供的 akg_agents 路径和环境名激活环境，创建独立测试目录
3. **编写测试脚本并运行** — 导入 Model / ModelNew，构造变异 shape/dtype 输入，逐 case 精度验证（子进程 + 超时保护）+ 性能测试，DevicePool (`from akg_agents.core.async_pool.device_pool import DevicePool`)支持多设备并行
4. **分析结果与生成报告** — 输出 JSON 原始数据 + Markdown 洞察分析（失败模式聚类、性能剖面、修复建议）

---

## Shape 变异策略

<shape-strategy>

### 总元素数分级

根据单个 tensor flatten 后的元素数分为三级：

| 级别 | 元素数 | 说明 |
|------|--------|------|
| 小 shape | ≤ 1e3 | 边界和极端情况 |
| 中等 shape | 1e4 ~ 1e7 | 常见业务场景 |
| 大 shape | ≥ 1e8 | 大规模计算，精度累积（按显存/内存量力而行） |

同一级别内 1 个代表值即可。

### 选值原则

1. **每级 1 个代表**：小、中、大各一个 shape
2. **对齐边界**：根据 kernel 的 BLOCK_SIZE / VECTOR_SIZE 取 BLOCK-1 和 BLOCK+1
3. **最小边界**：所有维度为 1
4. **极端纵横比**：某个维度为 1，其他维度大（如单 batch）
5. **非 2 的幂**：至少 1 个含素数或非对齐维度的 shape

### 典型 case 组成（8-12 个）

| # | 类型 | 说明 |
|---|------|------|
| 1 | 原始 shape | 基准 |
| 2 | 小 shape | ≤ 1e3 元素 |
| 3 | 中等 shape | 1e4 ~ 1e7 元素 |
| 4 | 大 shape | ≥ 1e8 元素（按显存/内存调整） |
| 5 | 最小边界 | 所有维度 = 1 |
| 6 | 极端纵横比 | 某维度 = 1 |
| 7 | 对齐边界 | BLOCK_SIZE ± 1 |
| 8 | 非 2 的幂 | 含素数维度 |
| 9+ | dtype 变异 | 原始 shape + 不同 dtype |

### 维度约束

- **自由维度**（如 batch_size）：直接变异输入 shape
- **参数绑定维度**（如 Linear 的 in_features）：需重新创建模型实例

</shape-strategy>

---

## dtype 变异策略

<dtype-strategy>

| 原始 dtype | 建议测试 |
|-----------|---------|
| float32 | float16, bfloat16 |
| bfloat16 | float16, float32 |
| float16 | bfloat16, float32 |

精度容忍度：

| dtype | 容忍度 |
|-------|-------|
| float32 | 0.02 |
| float16 | 0.004 |
| bfloat16 | 0.03 |

</dtype-strategy>

---

## 执行流程

### 阶段 0：信息收集

<phase0>

读取 task_code 和 kernel_code，提取：

1. **tensor 签名**：从 `get_inputs()` 和 `Model.forward()` 确定每个输入的 shape 和 dtype
2. **维度语义**：标注每个维度（batch、reduction_axis、spatial 等）
3. **参数约束**：从 `get_init_inputs()` 确定哪些维度受模型参数绑定
4. **kernel 分块参数**：查看 kernel 中的 BLOCK_SIZE、VECTOR_SIZE 等常量
5. **当前验证通过的配置**：shape、dtype、backend、dsl、arch

**示例**（矩阵乘法）：
```
算子: Standard_matrix_multiplication
配置: backend=cpu, dsl=cpp, arch=x86_64, framework=torch
tensor 签名: A=(M, K), B=(K, N) → Output=(M, N)
维度语义: M=行数, K=收缩维, N=列数
参数约束: get_init_inputs()=[] → 所有维度均为自由维度
kernel 参数: VECTOR_SIZE=8 (AVX2 float32) → N 对齐边界重要
当前通过: M=1024, K=4096, N=2048, dtype=float32
```

</phase0>

### 阶段 1：准备环境与测试目录

<phase1>

**1. 确认 akg_agents 信息**

向用户获取以下信息（如未提供则询问）：
- `AKG_AGENTS_PATH`：akg_agents 仓库路径（如 `~/akg/akg_agents`）
- `CONDA_ENV`：conda 环境名（如 `aikg`）

```bash
source <AKG_AGENTS_PATH>/env.sh
conda activate <CONDA_ENV>
```

**2. 创建测试目录**

在 `~/akg_agents_logs`目录下创建独立测试目录，命名为 `test_{op_name}_{YYYYMMDD_HHMMSS}_{4位随机ID}/`

```python
"""测试目录创建方法"""
import os, random, string
from datetime import datetime

def create_test_dir(op_name: str, base_dir: str = ".") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rid = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    path = os.path.join(base_dir, f"test_{op_name}_{timestamp}_{rid}")
    os.makedirs(path, exist_ok=True)
    return path
```

**目录产物**：

```
test_{op_name}_{timestamp}_{rid}/
├── robustness_test_{op_name}.py         # [脚本生成] 测试脚本
├── {op_name}_robustness_summary.json    # [脚本输出] 全部原始数据（JSON）
└── {op_name}_robustness_report.md       # [你撰写]   洞察分析报告（Markdown）
```

</phase1>

### 阶段 2：编写测试脚本并运行

<phase2>

生成一个独立的 Python 测试脚本，放入阶段 1 创建的目录。

**脚本职责边界**：脚本只负责运行测试并输出 `{op_name}_robustness_summary.json`。**禁止在脚本中生成 Markdown 报告**——Markdown 分析报告由你在阶段 3 亲自撰写。

脚本包含精度验证、性能测试，并输出输JSON 汇总。

**参考**：完整的测试脚本示例见 `@references/batch-robustness-test-example.py`。

```python
# 0. 导入必要模块和 akg_agents 库的 DevicePool
import sys, importlib, torch, time, json, asyncio
from akg_agents.core.async_pool.device_pool import DevicePool

VERIFY_TIMEOUT = 300  # 精度验证超时(秒)

# 1. 导入模型
sys.path.insert(0, VERIFY_DIR)
FrameworkModel = importlib.import_module(TASK_MODULE).Model
ModelNew = importlib.import_module(KERNEL_MODULE).ModelNew

# 2. 精度比对
def compare_outputs(fw_out, impl_out, limit): ...

# 3. 独立精度验证函数 (供子进程入口调用)
def verify_single(case_params, device_id):
    # 创建模型 → 构造输入 → 前向传播 → compare_outputs
    # 成功则正常返回，失败则 raise
    ...

# 4. 性能测试
def run_benchmark(model, inputs, warmup, run_times, backend):
    for _ in range(warmup):
        model(*inputs)
    # GPU: torch.cuda.synchronize()  NPU: torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(run_times):
        model(*inputs)
    # GPU: torch.cuda.synchronize()  NPU: torch.npu.synchronize()
    return (end - start) * 1e6 / run_times  # 微秒

# 5. 测试 case + 输入构造
TEST_CASES = [...]
def make_inputs(*shape_params, dtype, device): ...

# 6. 单 case 测试（精度子进程 + 性能 in-process）
async def test_single_case(case_idx, case, device_id):
    # 精度验证：以子进程运行，超时可 kill（含编译子进程）
    process = await asyncio.create_subprocess_exec(
        sys.executable, __file__, "--verify-case", str(case_idx), str(device_id),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, ...)
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=VERIFY_TIMEOUT)
        # returncode != 0 → FAIL, 跳过性能测试
    except asyncio.TimeoutError:
        process.kill()  # 终止子进程树（含编译进程）
        # → TIMEOUT, 跳过性能测试

    # 精度通过后，在当前进程内执行性能测试
    # run_benchmark(fw_model), run_benchmark(impl_model)
    # 输出：base_time_us, gen_time_us, speedup
    ...

# 7. AKG_agents 库 DevicePool 实现并行
pool = DevicePool(DEVICE_IDS)

async def run_all():
    async def run_on_device(idx, case):
        device_id = await pool.acquire_device()
        try:
            return await test_single_case(idx, case, device_id)
        finally:
            await pool.release_device(device_id)
    return await asyncio.gather(*[run_on_device(i, c) for i, c in enumerate(TEST_CASES)])

# 8. 入口：支持主模式和子进程模式
if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "--verify-case":
        # 子进程模式：执行单 case 精度验证，exit(0)=通过，exit(1)=失败
        verify_single(TEST_CASES[int(sys.argv[2])], int(sys.argv[3]))
    else:
        # 主模式：编排所有 case
        asyncio.run(run_all())
```

**关键点**：
- `make_inputs()` 按算子 tensor 签名编写，直接构造指定 shape/dtype 的 tensor
- 每个 case 调用两次 `make_inputs`（fw 和 impl 各一份），中间重置随机种子
- 性能测试遵循 akg_agents profile 脚本模式：warmup N 次 → 计时 M 次 → 取平均，结果以微秒输出
- **精度验证超时保护**：以子进程执行精度验证，`asyncio.wait_for` 控制超时，超时后 `process.kill()` 终止子进程及其启动的编译进程。不使用信号量（`signal.alarm`）——信号量无法杀死编译子进程
- 精度 FAIL 或 TIMEOUT 时跳过性能测试
- 脚本通过 `--verify-case` 参数复用自身作为子进程，避免代码重复

**运行**

```bash
cd test_{op_name}_{timestamp}_{rid}
python robustness_test_{op_name}.py

# 多设备并行:
AKG_AGENTS_DEVICES_LIST="0,1,2,3" python robustness_test_{op_name}.py
```

</phase2>

### 阶段 3：分析结果与撰写报告

<phase3>

脚本运行完成后产出 `{op_name}_robustness_summary.json`。

**你需要亲自完成以下工作**（不是由脚本完成）：

1. 读取 JSON 数据
2. 结合阶段 0 收集的算子信息（tensor 签名、kernel 分块参数等），对测试结果进行**人工分析**
3. 撰写 `{op_name}_robustness_report.md`（格式见下文报告规范）

这一步是你作为测试工程师的核心价值——根据具体的失败数据和算子特征做出有深度的归因判断，而不是机械地罗列数据。

**分析时关注的典型规律**：
- 所有非 2 的幂都失败 → kernel 对齐处理有 bug
- dim < BLOCK_SIZE 时崩溃 → 缺少小 shape 保护
- 特定 dtype 全部失败 → 类型转换路径有问题
- 大 shape 精度下降 → 浮点累积误差
- 特定 shape 性能显著退化 → 分块策略不适配
- 大 shape 编译超时 → kernel 编译复杂度随 shape 膨胀，需检查编译策略
- 多个 case 超时 → 编译缓存未命中或编译器本身存在性能问题

</phase3>

---

## 报告规范

<report>

### JSON 汇总 (`{op_name}_robustness_summary.json`)

包含全部原始数据，供程序消费和存档。结构：

```json
{
  "metadata": {
    "op_name": "ReLU",
    "test_date": "2026-02-26",
    "framework": "pytorch",
    "dsl": "triton_ascend",
    "backend": "npu",
    "arch": "Ascend910B",
    "original_shape": "(16, 16384)",
    "original_dtype": "torch.float32"
  },
  "summary": {
    "total_cases": 12,
    "accuracy_pass": 10,
    "accuracy_fail": 1,
    "accuracy_timeout": 1,
    "perf_pass": 5,
    "perf_warn": 3,
    "perf_fail": 2,
    "perf_skip": 2
  },
  "cases": [
    {
      "tag": "original",
      "shape": "(16, 16384)",
      "dtype": "torch.float32",
      "description": "原始通过 shape (中等，~262K 元素)",
      "device_id": 0,
      "accuracy": "PASS",
      "base_time_us": 85.07,
      "gen_time_us": 108.42,
      "speedup": 0.785,
      "perf_status": "FAIL",
      "error": null
    },
    {
      "tag": "large",
      "shape": "(16384, 16384)",
      "dtype": "torch.float32",
      "description": "大 shape (编译超时)",
      "device_id": 0,
      "accuracy": "TIMEOUT",
      "base_time_us": null,
      "gen_time_us": null,
      "speedup": null,
      "perf_status": "SKIP",
      "error": "精度验证超时 (300s)，已终止子进程"
    }
  ]
}
```

### Markdown 分析报告 (`{op_name}_robustness_report.md`)

基于 JSON 数据的**洞察分析**，不重复逐 case 列表。

```markdown
# 算子鲁棒性分析：{op_name}

**配置**: {framework} / {dsl} / {backend} / {arch}
**基准**: shape={original_shape}, dtype={original_dtype}
**原始数据**: `{op_name}_robustness_summary.json`

## 结论

{一句话定性判断，概括精度稳定性和性能主要风险。示例：}
精度在所有变异下均稳定；性能在半精度 dtype 和非对齐 shape 下存在显著退化，
根因指向 kernel 缺少 fp16/bf16 原生向量化路径和非对齐尾循环效率低。

## 失败模式

{将失败/警告 case 按共同根因聚类。}

### 模式 1: {名称，如"半精度 dtype 性能退化"}

- **涉及 case**: dtype_fp16 (0.18x), dtype_bf16 (0.17x)
- **归因**: kernel 内部将 fp16/bf16 转为 fp32 逐元素计算，丧失 SIMD 宽度优势。
- **风险**: 高 — 生产环境多使用 bf16 推理，直接影响端到端吞吐。

### 模式 2: ...

## 性能剖面

{描述 speedup 随规模/dtype 变化的趋势和拐点。}

| 规模区间 | 代表 shape | speedup 范围 | 趋势 |
|----------|-----------|-------------|------|
| ≤ 1e3 元素 | (4,64) | 0.89 ~ 1.65 | 调用开销主导，波动大 |
| 1e4 ~ 1e7 | (64,4096) | 0.53 ~ 0.94 | kernel 未体现优势 |
| ≥ 1e7 | (256,65536) | 1.10 | 大规模下优势显现 |

## 修复建议

{按优先级排列，每条附具体方向和预期收益。}

1. **[高] 添加 fp16/bf16 原生 kernel 分支** — 当前 0.17x → 目标 ≥ 1.0x
2. **[中] 优化中等规模 (1e4~1e7) 的分块策略** — 当前 0.53x，应至少持平
3. ...

## 已知限制（可接受）

- 极小 shape (1,1): kernel 调用开销 > 计算本身，speedup 波动属正常
```

</report>
---

## 反模式

<anti-patterns>

| 行为 | 说明 |
|------|------|
| 同一 size class 内重复测试 | 每级 1 个代表值即可 |
| batch 维大量枚举 | 3 个代表值（1, 2, 64）覆盖关键场景 |
| 只测 2 的幂 shape | 遗漏对齐边界 bug，需包含 BLOCK±1 |
| 在同一模型实例上变异参数绑定维度 | 会 shape mismatch，需重新创建模型 |
| 不固定随机种子 | 测试不可复现 |
| 把 kernel 逻辑复制为参考实现 | Model 必须用 PyTorch 原生实现 |
| 精度容忍度不区分 dtype | fp16/bf16/fp32 的合理误差差距很大 |
| 性能测试不做 warmup | 首次运行含编译/初始化开销，会偏高 |
| 改进建议留空 | 必须根据失败/警告类别生成具体建议，禁止只写标题无内容 |
| 文件夹命名不规范 | 必须使用 `test_{op_name}_{timestamp}_{random_id}/` 格式，确保唯一性和可追溯性 |
| 在脚本中生成 Markdown 报告 | Markdown 分析报告由你亲自撰写，不得在 Python 脚本中用字符串拼接生成 |
| 用信号量实现超时 | `signal.alarm` / 线程超时无法杀死编译子进程，必须用子进程 + `process.kill()` |
| 精度超时后仍执行性能测试 | 编译未完成意味着模型无法运行，性能测试无意义 |

</anti-patterns>
