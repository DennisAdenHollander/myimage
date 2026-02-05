#!/usr/bin/env python3

import re
from pathlib import Path


INPUT_FILES = [
    "sparkal1_class_stats.txt",
    "sparkal2_class_stats.txt",
]

OUTPUT_FILE = "merged_sorted_by_inst_score.txt"


def parse_stats_txt(path: str) -> list[dict]:
    rows = []

    line_re = re.compile(
        r"^\s*(\d+)\s+(.+?)\s+(\d+)\s+([0-9.]+)\s*$"
    )

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # sla headers, lege regels en scheidingslijnen over
            if (
                not line.strip()
                or line.startswith("Results")
                or line.startswith("-")
                or line.strip().startswith("ID")
            ):
                continue

            m = line_re.match(line)
            if not m:
                continue

            rows.append({
                "source": Path(path).stem,
                "class_id": int(m.group(1)),
                "class_name": m.group(2).strip(),
                "count": int(m.group(3)),
                "avg_score": float(m.group(4)),
            })

    return rows


def write_merged_txt(rows: list[dict], output_path: str) -> None:
    # kolombreedtes
    W_SRC = 16
    W_ID = 6
    W_NAME = 45
    W_COUNT = 10
    W_AVG = 14

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Merged class statistics (sorted by AVG_SCORE desc)\n\n")

        header = (
            f"{'SOURCE':<{W_SRC}}  "
            f"{'ID':>{W_ID}}  "
            f"{'CLASS':<{W_NAME}}  "
            f"{'COUNT':>{W_COUNT}}  "
            f"{'AVG_SCORE':>{W_AVG}}\n"
        )
        f.write(header)
        f.write("-" * len(header) + "\n")

        for r in rows:
            f.write(
                f"{r['source']:<{W_SRC}}  "
                f"{r['class_id']:>{W_ID}}  "
                f"{r['class_name']:<{W_NAME}}  "
                f"{r['count']:>{W_COUNT}}  "
                f"{r['avg_score']:>{W_AVG}.6f}\n"
            )


def main():
    all_rows = []

    for file in INPUT_FILES:
        all_rows.extend(parse_stats_txt(file))

    # sorteer op inst_score (hoog -> laag)
    all_rows.sort(key=lambda r: r["avg_score"], reverse=True)

    write_merged_txt(all_rows, OUTPUT_FILE)
    print(f"Wrote merged file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

