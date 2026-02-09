import csv
from pathlib import Path

AI_DIR = Path(__file__).parent.parent
OPTUNA_DIR = AI_DIR / "models" / "trained" / "optuna"

print("=" * 70)
print("OPTUNA TRIAL COMPARISON")
print("=" * 70)

results = []

for trial_dir in sorted(OPTUNA_DIR.glob("trial_*")):
    results_csv = trial_dir / "results.csv"
    args_yaml = trial_dir / "args.yaml"

    if not results_csv.exists():
        continue

    # Read last row of results.csv
    with open(results_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            continue
        last = rows[-1]

    # Find best mAP50-95 across all epochs
    best_map = max(float(r.get("   metrics/mAP50-95(B)", r.get("metrics/mAP50-95(B)", 0))) for r in rows)

    # Read key params from args.yaml
    params = {}
    with open(args_yaml, "r") as f:
        for line in f:
            for key in ["lr0:", "lrf:", "freeze:", "batch:"]:
                if line.startswith(key):
                    params[key.replace(":", "")] = line.split(":")[1].strip()

    results.append({
        "name": trial_dir.name,
        "best_map50_95": best_map,
        "params": params,
    })

# Sort by best mAP50-95
results.sort(key=lambda x: x["best_map50_95"], reverse=True)

print(f"\n{'Trial':<12} {'mAP50-95':<12} {'lr0':<20} {'freeze':<8} {'batch':<8} {'lrf'}")
print("-" * 70)
for r in results:
    p = r["params"]
    print(f"{r['name']:<12} {r['best_map50_95']:<12.4f} {p.get('lr0','?'):<20} {p.get('freeze','?'):<8} {p.get('batch','?'):<8} {p.get('lrf','?')}")

if results:
    best = results[0]
    print(f"\nBest: {best['name']} with mAP50-95 = {best['best_map50_95']:.4f}")
    print(f"Model: {OPTUNA_DIR / best['name'] / 'weights' / 'best.pt'}")