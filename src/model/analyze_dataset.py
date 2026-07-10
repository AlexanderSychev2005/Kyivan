"""
Kyivan Dataset Analyzer.

This script reads the aggregated JSONL dataset and generates a comprehensive
statistical report for each source. It details the exact distribution of
dialects and dating formats present, which is crucial for building the final
probability mappings (KL-Divergence arrays) in the next pipeline stage.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict


def print_top_counter(counter: Counter, top_n: int = 15, indent: str = "    ") -> None:
    """Helper function to print the top items in a Counter."""
    if not counter:
        print(f"{indent}None found")
        return

    total = sum(counter.values())
    for item, count in counter.most_common(top_n):
        percentage = (count / total) * 100
        print(f"{indent}{str(item):<40} {count:>8,} ({percentage:>5.1f}%)")

    if len(counter) > top_n:
        remaining = len(counter) - top_n
        print(f"{indent}... and {remaining} more unique values")


def main() -> None:
    """Parses arguments and generates the statistical report."""
    parser = argparse.ArgumentParser(description="Analyze JSONL Dataset Metadata.")
    parser.add_argument(
        "--input", default="dataset.jsonl", help="Path to the aggregated JSONL dataset"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: File not found -> {input_path}")
        return

    # Initialize data structures
    # stats[source] = {"dialects": Counter, "dates": Counter, "total": int}
    stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"dialects": Counter(), "dates": Counter(), "total": 0}
    )

    print(f"⏳ Reading dataset {input_path}...\n")

    total_lines = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            source = record.get("source", "UNKNOWN")
            dialect = record.get("dialect")
            date = record.get("date")

            stats[source]["total"] += 1
            if dialect is not None:
                stats[source]["dialects"][str(dialect)] += 1
            if date is not None:
                stats[source]["dates"][str(date)] += 1

            total_lines += 1

    print("=" * 80)
    print(f"DATASET ANALYSIS REPORT (Total Records: {total_lines:,})")
    print("=" * 80)

    for source in sorted(stats.keys()):
        data = stats[source]
        total = data["total"]
        n_dialects = sum(data["dialects"].values())
        n_dates = sum(data["dates"].values())

        print(f"\n📂 SOURCE: [{source.upper()}] - {total:,} records")
        print("-" * 80)

        # Dialect Distribution
        coverage_dialect = (n_dialects / total) * 100 if total > 0 else 0
        print(f"🗣️  DIALECTS (Coverage: {coverage_dialect:.1f}%)")
        print_top_counter(data["dialects"], top_n=10)

        # Date Distribution
        coverage_date = (n_dates / total) * 100 if total > 0 else 0
        print(f"\n📅 DATES (Coverage: {coverage_date:.1f}%)")
        print_top_counter(data["dates"], top_n=10)

        print("=" * 80)


if __name__ == "__main__":
    main()
