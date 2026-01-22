
"""
run_compare.py
Runs Random (k trials), Greedy, and Greedy+SA on the same dataset and writes a single compare_summary.json.
Pure stdlib; calls main_sa.py via subprocess so you don't need to modify your main script.
"""

import argparse, json, subprocess, sys
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent

def run_once(helpers, tickets, scheduler, out_prefix, slot_minutes, time_policy, seed,
             sa_steps, sa_alpha, sa_beta, sa_T0, sa_cooling):
    out_csv = HERE / f"{out_prefix}_{scheduler}.csv"
    out_json = HERE / f"{out_prefix}_{scheduler}.json"
    cmd = [
        sys.executable, str(HERE / "main_sa.py"),
        "--helpers", helpers,
        "--tickets", tickets,
        "--scheduler", scheduler,
        "--slot_minutes", str(slot_minutes),
        "--time_policy", time_policy,
        "--out_csv", str(out_csv),
        "--out_json", str(out_json),
        "--sa_steps", str(sa_steps),
        "--sa_alpha", str(sa_alpha),
        "--sa_beta", str(sa_beta),
        "--sa_T0", str(sa_T0),
        "--sa_cooling", str(sa_cooling),
        "--seed", str(seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
    if proc.returncode != 0:
        raise RuntimeError(f"{scheduler} run failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    data = json.loads(out_json.read_text(encoding="utf-8"))
    return data

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--helpers", required=True)
    ap.add_argument("--tickets", required=True)
    ap.add_argument("--slot_minutes", type=int, default=15)
    ap.add_argument("--time_policy", choices=["finish_sla","start_sla"], default="finish_sla")
    ap.add_argument("--random_trials", type=int, default=10, help="how many random runs to average")
    ap.add_argument("--seed", type=int, default=1337)
    # SA params
    ap.add_argument("--sa_steps", type=int, default=5000)
    ap.add_argument("--sa_alpha", type=float, default=100.0)
    ap.add_argument("--sa_beta", type=float, default=1.0)
    ap.add_argument("--sa_T0", type=float, default=1.0)
    ap.add_argument("--sa_cooling", type=float, default=0.999)
    ap.add_argument("--out_json", default="compare_summary.json")
    return ap.parse_args()

def metric_row(tag, data):
    mo = data["metrics_overall"]
    return {
        "algo": tag,
        "sla_breach_rate": mo["sla_breach_rate"],
        "total_tardiness_minutes": mo["total_tardiness_minutes"],
        "workload_variance": mo["workload_variance"],
        "max_helper_load": max((h["tickets"] for h in data.get("metrics_by_helper", [])), default=0),
    }

def main():
    args = parse_args()
    helpers = str(Path(args.helpers))
    tickets = str(Path(args.tickets))

    # Greedy once
    greedy = run_once(helpers, tickets, "greedy", "cmp", args.slot_minutes, args.time_policy,
                      args.seed, args.sa_steps, args.sa_alpha, args.sa_beta, args.sa_T0, args.sa_cooling)

    # Greedy+SA once
    sa = run_once(helpers, tickets, "greedy+sa", "cmp", args.slot_minutes, args.time_policy,
                  args.seed, args.sa_steps, args.sa_alpha, args.sa_beta, args.sa_T0, args.sa_cooling)

    # Random k trials (different seeds)
    random_rows = []
    for i in range(args.random_trials):
        rseed = args.seed + i
        rdata = run_once(helpers, tickets, "random", f"cmp_r{i}", args.slot_minutes, args.time_policy,
                         rseed, args.sa_steps, args.sa_alpha, args.sa_beta, args.sa_T0, args.sa_cooling)
        random_rows.append(metric_row("random", rdata))

    # Aggregate random stats (mean ± std)
    def agg(field):
        vals = [row[field] for row in random_rows]
        mean = sum(vals)/len(vals) if vals else 0.0
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return mean, std

    r_mean_breach, r_std_breach = agg("sla_breach_rate")
    r_mean_tard, r_std_tard = agg("total_tardiness_minutes")
    r_mean_var, r_std_var = agg("workload_variance")
    r_mean_max,  r_std_max  = agg("max_helper_load")

    out = {
        "dataset": {
            "helpers_csv": helpers,
            "tickets_csv": tickets,
            "slot_minutes": args.slot_minutes,
            "time_policy": args.time_policy
        },
        "sa_params": {
            "alpha_sla": args.sa_alpha,
            "beta_variance": args.sa_beta,
            "steps": args.sa_steps,
            "T0": args.sa_T0,
            "cooling": args.sa_cooling
        },
        "random_trials": args.random_trials,
        "results": {
            "random_mean": {
                "sla_breach_rate": r_mean_breach,
                "sla_breach_rate_std": r_std_breach,
                "total_tardiness_minutes": r_mean_tard,
                "total_tardiness_minutes_std": r_std_tard,
                "workload_variance": r_mean_var,
                "workload_variance_std": r_std_var,
                "max_helper_load": r_mean_max,
                "max_helper_load_std": r_std_max
            },
            "greedy": metric_row("greedy", greedy),
            "greedy+sa": metric_row("greedy+sa", sa)
        },
        "notes": "All methods share the same constraints. Random uses qualified helpers uniformly. Seeds varied across random trials."
    }

    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json}")
    print("Random (mean±std) vs Greedy vs Greedy+SA summary ready for your slides.")

if __name__ == "__main__":
    main()
