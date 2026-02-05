#!/usr/bin/env python3

import re
import pandas as pd


N_CLASSES = 150
CLASSES_FILE = "classes.txt"

INPUT_FILES = [
    ("sparkal1_instance_class_scores.csv", "sparkal1_class_stats.txt"),
    ("sparkal2_instance_class_scores.csv", "sparkal2_class_stats.txt"),
]


def load_classes_table(path: str) -> dict[int, str]:
    """Parse classes.txt: 'index | class name'"""
    class_map: dict[int, str] = {}
    line_re = re.compile(r"^\s*(\d+)\s*\|\s*(.*?)\s*$")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = line_re.match(line)
            if not m:
                continue
            idx = int(m.group(1))
            name = m.group(2).strip()
            class_map[idx] = name

    return class_map


def analyze_csv(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    result = (
        df.groupby("instance_class_id")
        .agg(
            count=("inst_score", "size"),
            avg_inst_score=("inst_score", "mean"),
        )
        .reindex(range(N_CLASSES), fill_value=0)
        .reset_index()
        .rename(columns={"index": "instance_class_id"})
    )
    return result


def write_txt(result, class_map, input_csv, output_txt):
    # kolombreedtes
    W_ID = 6
    W_NAME = 45
    W_COUNT = 10
    W_AVG = 14

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"Results for: {input_csv}\n\n")

        header = (
            f"{'ID':>{W_ID}}  "
            f"{'CLASS':<{W_NAME}}  "
            f"{'COUNT':>{W_COUNT}}  "
            f"{'AVG_SCORE':>{W_AVG}}\n"
        )
        f.write(header)
        f.write("-" * len(header) + "\n")

        for _, row in result.iterrows():
            cid = int(row["instance_class_id"])
            name = class_map[cid]
            count = int(row["count"])
            avg = float(row["avg_inst_score"])

            f.write(
                f"{cid:>{W_ID}}  "
                f"{name:<{W_NAME}}  "
                f"{count:>{W_COUNT}}  "
                f"{avg:>{W_AVG}.6f}\n"
            )


def main():
    class_map = load_classes_table(CLASSES_FILE)

    for input_csv, output_txt in INPUT_FILES:
        result = analyze_csv(input_csv)
        write_txt(result, class_map, input_csv, output_txt)
        print(f"Wrote {output_txt}")


if __name__ == "__main__":
    main()

