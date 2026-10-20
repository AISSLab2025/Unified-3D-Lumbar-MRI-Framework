"""Summarize complete patient runs from the persistent timing log."""
import argparse
import json
import statistics
from pathlib import Path

from stage_timing import CV_STAGES


def summarize(records, skip_first=0):
    considered = records[skip_first:]
    accepted = [record for record in considered if record["status"] == "success"
                and all(record["stages"].get(name, {}).get("status") == "success"
                        for name in CV_STAGES)]
    def stats(values):
        return {
            "count": len(values),
            "mean_seconds": statistics.mean(values) if values else None,
            "std_seconds": statistics.stdev(values) if len(values) > 1 else None,
            "min_seconds": min(values) if values else None,
            "max_seconds": max(values) if values else None,
        }
    return {
        "units": "seconds",
        "runs_in_log": len(records),
        "initial_runs_excluded": min(skip_first, len(records)),
        "complete_runs_used": len(accepted),
        "failed_or_partial_runs_excluded": len(considered) - len(accepted),
        "stages": {name: stats([row["stages"][name]["seconds"] for row in accepted])
                   for name in CV_STAGES},
        "cv_total": stats([row["cv_total_seconds"] for row in accepted]),
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "benchmark" / "stage_timings.jsonl")
    parser.add_argument("--output", type=Path, default=root / "benchmark" / "timing_summary.json")
    parser.add_argument("--skip-first", type=int, default=0, help="Exclude this many initial records, e.g. a warm-up patient.")
    args = parser.parse_args()
    if args.skip_first < 0:
        parser.error("--skip-first must be nonnegative")
    with args.input.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    report = summarize(records, args.skip_first)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
