"""
算子鲁棒性测试 — 矩阵乘法 (Standard_matrix_multiplication)

基于 akg_agents 已验证通过的 case，直接导入 Model / ModelNew，
构造不同 shape/dtype 的输入进行精度验证和性能对比。
使用 akg_agents 库的 DevicePool 支持多设备并行。
精度验证以子进程方式执行，支持超时终止（含编译子进程）。

已验证配置: M=1024, K=4096, N=2048, dtype=float32, backend=cpu, dsl=cpp

使用方法:
    source <AKG_AGENTS_PATH>/env.sh && conda activate <CONDA_ENV>
    python robustness_test_matmul.py

    # 多设备并行:
    AKG_AGENTS_DEVICES_LIST="0,1,2,3" python robustness_test_matmul.py
"""

import sys
import os
import time
import json
import importlib
import asyncio
import torch

from akg_agents.core.async_pool.device_pool import DevicePool


# ============================================================
# 配置 — 根据实际 case 修改
# ============================================================

# 已通过验证的 case 目录
VERIFY_DIR = os.path.expanduser(
    "~/akg_agents_logs/Task_ih0at2l3/passed_cases/"
    "akg_agents_kernelbench_2_Standard_matrix_multiplication_/"
    "Iteration0_Step01_verify"
)
TASK_MODULE = "akg_agents_kernelbench_2_Standard_matrix_multiplication__torch"
KERNEL_MODULE = "akg_agents_kernelbench_2_Standard_matrix_multiplication__cpp_impl"

OP_NAME = "Standard_matrix_multiplication"
FRAMEWORK = "pytorch"
DSL = "cpp"
BACKEND = "cpu"       # "ascend" / "cuda" / "cpu"
ARCH = "x86_64"       # "Ascend910B" / "sm_80" / "x86_64"
DEVICE_IDS = [0]      # 多设备并行: [0, 1, 2, 3]
SEED = 42
WARMUP_TIMES = 5
RUN_TIMES = 50
VERIFY_TIMEOUT = 300  # 精度验证超时(秒)，超时则终止子进程并跳过性能测试


# ============================================================
# 导入模型
# ============================================================

sys.path.insert(0, VERIFY_DIR)

task_mod = importlib.import_module(TASK_MODULE)
FrameworkModel = task_mod.Model
get_init_inputs = task_mod.get_init_inputs

try:
    kernel_mod = importlib.import_module(KERNEL_MODULE)
    ModelNew = kernel_mod.ModelNew
except Exception as e:
    print(f"ERROR: 无法导入 kernel_code: {e}")
    print("请检查编译环境和依赖（如 C++ 编译器、CUDA toolkit 等）")
    sys.exit(1)


# ============================================================
# 精度比对
# ============================================================

def get_limit(dtype):
    """根据 dtype 返回误差容忍度"""
    return {
        torch.float16: 0.004,
        torch.bfloat16: 0.03,
        torch.int8: 0.01,
    }.get(dtype, 0.02)


def compare_outputs(fw_out, impl_out, limit):
    """
    比对精度，判定标准与 akg_agents verify 模板一致：
    相对误差超过 limit 的元素个数 <= 总元素数 * limit
    """
    fw_flat = fw_out.flatten().detach().cpu().float()
    impl_flat = impl_out.flatten().detach().cpu().float()

    if fw_flat.shape != impl_flat.shape:
        raise AssertionError(
            f"输出 shape 不一致: framework={fw_out.shape}, impl={impl_out.shape}"
        )

    abs_diff = torch.abs(fw_flat - impl_flat)
    abs_ref = torch.abs(fw_flat).clamp(min=1e-8)
    rel_err = abs_diff / abs_ref

    err_count = (rel_err > limit).sum().item()
    max_allowed = int(fw_flat.numel() * limit)

    if err_count > max_allowed:
        max_err = rel_err.max().item()
        mean_err = rel_err.mean().item()
        raise AssertionError(
            f"精度不达标: 超限元素 {err_count}/{max_allowed}, "
            f"max_rel_err={max_err:.6e}, mean_rel_err={mean_err:.6e}"
        )


def verify_single(M, K, N, dtype, device_id):
    """
    独立精度验证函数。
    通过 --verify-case 作为子进程调用，支持超时终止。
    成功时 exit(0)，失败时 exit(1) 并输出错误到 stderr。
    """
    device = get_device(device_id)
    init_params = get_init_inputs()

    torch.manual_seed(SEED)
    fw_model = FrameworkModel(*init_params)
    impl_model = ModelNew(*init_params)

    if device.type != "cpu":
        fw_model = fw_model.to(device)
        impl_model = impl_model.to(device)

    torch.manual_seed(SEED)
    inputs_fw = make_matmul_inputs(M, K, N, dtype, device)
    torch.manual_seed(SEED)
    inputs_impl = make_matmul_inputs(M, K, N, dtype, device)

    fw_out = fw_model(*inputs_fw)
    impl_out = impl_model(*inputs_impl)

    fw_outs = fw_out if isinstance(fw_out, (list, tuple)) else [fw_out]
    impl_outs = impl_out if isinstance(impl_out, (list, tuple)) else [impl_out]

    if len(fw_outs) != len(impl_outs):
        raise AssertionError(
            f"输出个数不一致: framework={len(fw_outs)}, impl={len(impl_outs)}"
        )

    limit = get_limit(dtype)
    for fw_o, impl_o in zip(fw_outs, impl_outs):
        compare_outputs(fw_o, impl_o, limit)


# ============================================================
# 性能测试
# ============================================================


def run_benchmark(model, inputs, warmup=WARMUP_TIMES, run_times=RUN_TIMES):
    """
    性能基准测试:
    warmup N 次 → 计时 M 次 → 取平均。
    结果单位: 微秒 (us)。

    GPU/NPU 场景需在 warmup 和计时循环后加 synchronize:
      - CUDA:  torch.cuda.synchronize()
      - Ascend: torch.npu.synchronize()
    """
    for _ in range(warmup):
        model(*inputs)
    if BACKEND == "cuda":
        torch.cuda.synchronize()
    elif BACKEND == "ascend":
        torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(run_times):
        model(*inputs)
    if BACKEND == "cuda":
        torch.cuda.synchronize()
    elif BACKEND == "ascend":
        torch.npu.synchronize()
    end = time.perf_counter()

    execution_time_us = (end - start) * 1e6 / run_times
    return execution_time_us


# ============================================================
# 测试 case 定义
# ============================================================

# 算子分析:
#   forward(A, B) -> torch.matmul(A, B)
#   A=(M, K), B=(K, N), Output=(M, N)
#   get_init_inputs() = [] -> 所有维度均为自由维度
#   kernel: AVX2 float32, VECTOR_SIZE=8 -> N 对齐边界重要
#
# 元素数分级 (以最大 tensor 元素数计):
#   小 <= 1e3, 中 1e4~1e7, 大 >= 1e8

TEST_CASES = [
    # (tag,          M,     K,     N,     dtype,          description)
    ("original",     1024,  4096,  2048,  torch.float32,  "原始通过 shape (中等, max ~4M 元素)"),
    ("small",        4,     16,    4,     torch.float32,  "小 (max tensor 64 元素)"),
    ("medium",       128,   512,   256,   torch.float32,  "中等 (max tensor ~64K 元素)"),
    ("large",        4096,  4096,  4096,  torch.float32,  "大 (max tensor ~16M 元素)"),
    ("min_edge",     1,     1,     1,     torch.float32,  "最小边界: 所有维度=1"),
    ("single_row",   1,     4096,  2048,  torch.float32,  "单行 batch (极端纵横比)"),
    ("non_align_N",  64,    128,   127,   torch.float32,  "N=127, 非 VECTOR_SIZE(8) 对齐"),
    ("dtype_fp16",   1024,  4096,  2048,  torch.float16,  "dtype 变异: float16"),
]


def make_matmul_inputs(M, K, N, dtype, device):
    """根据 matmul 的 tensor 签名构造输入: A=(M,K), B=(K,N)"""
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    return [A, B]


def get_device(device_id):
    """根据 BACKEND 和 device_id 返回 torch.device"""
    if BACKEND == "ascend":
        return torch.device(f"npu:{device_id}")
    elif BACKEND == "cuda":
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


# ============================================================
# 单 case 测试 (精度验证 + 性能测试)
# ============================================================
#
# 精度验证以子进程方式运行，原因：
#   算子首次前向传播会触发编译（C++/Triton/AscendC），编译过程会启动
#   新的子进程。如果使用信号量（signal.alarm）或线程超时，只能终止当前
#   进程/线程，无法杀死编译子进程，导致编译卡死。
#   通过 asyncio.create_subprocess_exec 启动子进程并用 asyncio.wait_for
#   控制超时，超时后 process.kill() 可终止整个子进程树。
#


async def test_single_case(case_idx, tag, M, K, N, dtype, desc, device_id):
    """对单个 shape/dtype 执行精度验证和性能测试"""
    device = get_device(device_id)

    result = {
        "tag": tag,
        "shape": f"({M}, {K}, {N})",
        "dtype": str(dtype),
        "description": desc,
        "device_id": device_id,
        "accuracy": None,
        "base_time_us": None,
        "gen_time_us": None,
        "speedup": None,
        "perf_status": None,
        "error": None,
    }

    # --- 精度验证 (子进程 + 超时保护) ---
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    script_path = os.path.abspath(__file__)
    process = await asyncio.create_subprocess_exec(
        sys.executable, script_path,
        "--verify-case", str(case_idx), str(device_id),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=VERIFY_TIMEOUT
        )

        if process.returncode != 0:
            result["accuracy"] = "FAIL"
            result["error"] = stderr_bytes.decode(errors="replace").strip()
            result["perf_status"] = "SKIP"
            return result

        result["accuracy"] = "PASS"

    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        result["accuracy"] = "TIMEOUT"
        result["error"] = f"精度验证超时 ({VERIFY_TIMEOUT}s)，已终止子进程"
        result["perf_status"] = "SKIP"
        return result

    # --- 性能测试 (精度已通过，在当前进程内执行) ---
    try:
        init_params = get_init_inputs()
        torch.manual_seed(SEED)
        fw_model = FrameworkModel(*init_params)
        impl_model = ModelNew(*init_params)

        if device.type != "cpu":
            fw_model = fw_model.to(device)
            impl_model = impl_model.to(device)

        torch.manual_seed(SEED)
        inputs_perf = make_matmul_inputs(M, K, N, dtype, device)

        base_time = run_benchmark(fw_model, inputs_perf)
        gen_time = run_benchmark(impl_model, inputs_perf)
        speedup = base_time / gen_time if gen_time > 0 else float("inf")

        result["base_time_us"] = round(base_time, 2)
        result["gen_time_us"] = round(gen_time, 2)
        result["speedup"] = round(speedup, 3)

        if speedup >= 0.95:
            result["perf_status"] = "PASS"
        elif speedup >= 0.8:
            result["perf_status"] = "WARN"
        else:
            result["perf_status"] = "FAIL"

    except Exception as e:
        result["base_time_us"] = None
        result["gen_time_us"] = None
        result["speedup"] = None
        result["perf_status"] = "ERROR"
        result["error"] = str(e)

    return result


# ============================================================
# DevicePool 并行执行
# ============================================================


async def run_all_tests():
    """
    使用 akg_agents 库的 DevicePool 并行分配设备，执行所有测试 case。
    单设备时退化为顺序执行。
    """
    pool = DevicePool(DEVICE_IDS)

    async def run_on_device(case_idx, case):
        device_id = await pool.acquire_device()
        try:
            tag, M, K, N, dtype, desc = case
            return await test_single_case(case_idx, tag, M, K, N, dtype, desc, device_id)
        finally:
            await pool.release_device(device_id)

    tasks = [run_on_device(idx, case) for idx, case in enumerate(TEST_CASES)]
    results = await asyncio.gather(*tasks)
    return list(results)


# ============================================================
# 结果输出
# ============================================================


def print_results(results):
    """打印并保存测试结果"""
    header = (
        f"{'Tag':15s} {'Shape':25s} {'dtype':12s} "
        f"{'Acc':6s} {'base(us)':>10s} {'gen(us)':>10s} {'speedup':>8s} {'Perf':6s}"
    )
    print(f"\n{header}")
    print("-" * 96)

    for r in results:
        base = f"{r['base_time_us']:>10.2f}" if r.get("base_time_us") is not None else "       N/A"
        gen = f"{r['gen_time_us']:>10.2f}" if r.get("gen_time_us") is not None else "       N/A"
        spd = f"{r['speedup']:>7.3f}x" if r.get("speedup") is not None else "     N/A"
        perf = r.get("perf_status", "N/A")

        print(
            f"{r['tag']:15s} {r['shape']:25s} {r['dtype']:12s} "
            f"{r['accuracy']:6s} {base} {gen} {spd} {perf:6s}"
        )

        if r["accuracy"] in ("FAIL", "TIMEOUT"):
            print(f"  -> {r.get('error', '')}")

    total = len(results)
    acc_passed = sum(1 for r in results if r["accuracy"] == "PASS")
    acc_fail = sum(1 for r in results if r["accuracy"] == "FAIL")
    acc_timeout = sum(1 for r in results if r["accuracy"] == "TIMEOUT")
    print("-" * 96)
    status_parts = [f"PASS {acc_passed}"]
    if acc_fail > 0:
        status_parts.append(f"FAIL {acc_fail}")
    if acc_timeout > 0:
        status_parts.append(f"TIMEOUT {acc_timeout}")
    print(f"精度: {' / '.join(status_parts)} (共 {total})")

    perf_pass = sum(1 for r in results if r.get("perf_status") == "PASS")
    perf_warn = sum(1 for r in results if r.get("perf_status") == "WARN")
    perf_fail = sum(1 for r in results if r.get("perf_status") == "FAIL")
    perf_skip = sum(1 for r in results if r.get("perf_status") == "SKIP")
    perf_tested = perf_pass + perf_warn + perf_fail
    if perf_tested > 0:
        print(f"性能通过率 (speedup >= 0.95): {perf_pass}/{perf_tested}")

    # 保存结构化 JSON
    original = TEST_CASES[0]
    summary = {
        "metadata": {
            "op_name": OP_NAME,
            "test_date": time.strftime("%Y-%m-%d"),
            "framework": FRAMEWORK,
            "dsl": DSL,
            "backend": BACKEND,
            "arch": ARCH,
            "original_shape": f"({original[1]}, {original[2]}, {original[3]})",
            "original_dtype": str(original[4]),
        },
        "summary": {
            "total_cases": total,
            "accuracy_pass": acc_passed,
            "accuracy_fail": acc_fail,
            "accuracy_timeout": acc_timeout,
            "perf_pass": perf_pass,
            "perf_warn": perf_warn,
            "perf_fail": perf_fail,
            "perf_skip": perf_skip,
        },
        "cases": results,
    }

    output_path = f"{OP_NAME}_robustness_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"详细结果: {output_path}")


# ============================================================
# 入口
# ============================================================


if __name__ == "__main__":
    # --verify-case <idx> [device_id]: 子进程模式，执行单 case 精度验证
    # 主进程通过此入口启动子进程来实现超时保护
    if len(sys.argv) > 1 and sys.argv[1] == "--verify-case":
        case_idx = int(sys.argv[2])
        device_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        tag, M, K, N, dtype, desc = TEST_CASES[case_idx]
        try:
            verify_single(M, K, N, dtype, device_id)
        except Exception as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    # 主模式：执行全部测试
    original = TEST_CASES[0]
    print("=" * 96)
    print(f"算子鲁棒性测试: {OP_NAME}")
    print(f"配置: framework={FRAMEWORK}, dsl={DSL}, backend={BACKEND}, arch={ARCH}")
    print(f"原始通过: shape=({original[1]}, {original[2]}, {original[3]}), dtype={original[4]}")
    print(f"设备: {DEVICE_IDS} (共 {len(DEVICE_IDS)} 个)")
    print(f"精度验证超时: {VERIFY_TIMEOUT}s")
    print("=" * 96)

    results = asyncio.run(run_all_tests())
    print_results(results)
