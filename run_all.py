"""
一键调度脚本: 跑完全部 6 configs × 3 seeds = 18 次训练 + 回测
=============================================================

用法 (挂机过夜):
  python run_all.py

特性:
  - 每组训练完立即做回测, 结果立刻落盘
  - 断点续跑: 如果某个 (config, seed) 的回测 CSV 已存在就跳过
  - 运行进度保存在 results/run_progress.json, 崩了重启会继续
  - 每组预计 1.5-2 小时 (400k steps @ CPU), 总 30-36 小时
  - 跑完自动调用 aggregate_seeds.py --all 生成最终聚合表

防御机制:
  - 单组崩了不影响其他组 (try/except)
  - 日志同时写到控制台和 run_all.log
  - 键盘 Ctrl+C 会 graceful 退出, 下次重启从同一位置续跑

文件后缀:
  每组产出 models/portfolio_ppo_<config>_v5_ppo_s<seed>_best.zip
  以及相应的 results/*_v5_ppo_s<seed>.*
  聚合后 results/backtest_metrics_aggregated_<config>_v5_ppo.csv
  全局汇总 results/backtest_metrics_summary_ALL_v5_ppo.csv
"""
import os
import sys
import json
import time
import subprocess
import datetime
from pathlib import Path

CONFIGS = ['HKUST_Full', 'HKUST_NoAttn',   # 旗舰优先
           'EODHD_Full', 'EODHD_NoAttn',
           'NONE_NoSent', 'NONE_Vanilla',
           'HKUST_Full_CGS', 'HKUST_NoAttn_CGS']  # CGS novelty
SEEDS = [42, 123, 2024]

TIMESTEPS = 400_000
EVAL_FREQ = 50_000
VAL_DAYS = 75

PROGRESS_FILE = "results/run_progress.json"
LOG_FILE = "results/run_all.log"


def log(msg):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"done": [], "failed": []}


def save_progress(p):
    os.makedirs("results", exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)


def has_result(config, seed):
    """判断该 (config, seed) 的最终回测 CSV 是否已存在"""
    path = f"results/backtest_metrics_summary_{config}_v5_ppo_s{seed}.csv"
    return os.path.exists(path)


def run_one(config, seed):
    """跑一次 train + backtest"""
    key = f"{config}_s{seed}"

    # 跳过已完成
    if has_result(config, seed):
        log(f"[skip] {key} 已存在回测结果, 跳过")
        return "skip"

    log(f"[START] {key} 开始训练")
    t0 = time.time()

    # ---- 训练 ----
    cmd_train = [
        sys.executable, "multi_train_ppo.py",
        "--config", config,
        "--seed", str(seed),
        "--timesteps", str(TIMESTEPS),
        "--eval_freq", str(EVAL_FREQ),
        "--val_days", str(VAL_DAYS),
    ]
    try:
        ret = subprocess.run(cmd_train, check=False)
        if ret.returncode != 0:
            log(f"[FAIL] {key} 训练返回码 {ret.returncode}")
            return "fail_train"
    except KeyboardInterrupt:
        log(f"[INTERRUPT] {key} 训练被 Ctrl+C 打断, 不计入 done")
        raise
    except Exception as e:
        log(f"[FAIL] {key} 训练异常: {e}")
        return "fail_train"

    train_elapsed = time.time() - t0
    log(f"[TRAIN OK] {key} 训练耗时 {train_elapsed/60:.1f} 分钟")

    # ---- 回测 ----
    cmd_bt = [
        sys.executable, "multi_backtest_ppo.py",
        "--config", config,
        "--seed", str(seed),
    ]
    try:
        ret = subprocess.run(cmd_bt, check=False)
        if ret.returncode != 0:
            log(f"[FAIL] {key} 回测返回码 {ret.returncode}")
            return "fail_bt"
    except KeyboardInterrupt:
        log(f"[INTERRUPT] {key} 回测被 Ctrl+C 打断")
        raise
    except Exception as e:
        log(f"[FAIL] {key} 回测异常: {e}")
        return "fail_bt"

    total_elapsed = time.time() - t0
    log(f"[DONE] {key} 总耗时 {total_elapsed/60:.1f} 分钟 "
        f"({total_elapsed/3600:.2f} 小时)")
    return "done"


def main():
    os.makedirs("results", exist_ok=True)
    progress = load_progress()

    log("=" * 70)
    log(f"开始全套 PPO 训练: {len(CONFIGS)} configs × {len(SEEDS)} seeds = "
        f"{len(CONFIGS) * len(SEEDS)} 次")
    log(f"已完成: {len(progress['done'])}, 失败: {len(progress['failed'])}")
    log("=" * 70)

    total = len(CONFIGS) * len(SEEDS)
    idx = 0
    run_t0 = time.time()

    for config in CONFIGS:
        for seed in SEEDS:
            idx += 1
            key = f"{config}_s{seed}"
            log(f"\n\n--- [{idx}/{total}] {key} ---")

            if key in progress["done"]:
                log(f"[已记录为 done] 跳过")
                continue

            try:
                result = run_one(config, seed)
            except KeyboardInterrupt:
                log("用户中断, 保存进度后退出")
                save_progress(progress)
                log(f"下次运行会从 {key} 继续")
                sys.exit(0)

            if result == "done" or result == "skip":
                if key not in progress["done"]:
                    progress["done"].append(key)
                if key in progress["failed"]:
                    progress["failed"].remove(key)
            else:
                if key not in progress["failed"]:
                    progress["failed"].append(key)

            save_progress(progress)

            # 打印整体进度
            done_n = len(progress["done"])
            elapsed = time.time() - run_t0
            if done_n > 0:
                avg_per_task = elapsed / done_n
                remaining = (total - done_n) * avg_per_task
                log(f"总进度: {done_n}/{total} | "
                    f"已跑 {elapsed/3600:.1f} h | "
                    f"预计还需 {remaining/3600:.1f} h")

    log("\n\n" + "=" * 70)
    log(f"全部完成! done={len(progress['done'])}, failed={len(progress['failed'])}")
    if progress["failed"]:
        log(f"失败列表: {progress['failed']}")
    log("=" * 70)

    # 聚合
    log("\n开始聚合 3-seed 结果...")
    try:
        subprocess.run([sys.executable, "aggregate_seeds.py", "--all"], check=False)
    except Exception as e:
        log(f"聚合失败: {e}")

    log(f"\n大总表: results/backtest_metrics_summary_ALL_v5_ppo.csv")
    log("挂机完成, 请睡个好觉 😴")


if __name__ == "__main__":
    main()
