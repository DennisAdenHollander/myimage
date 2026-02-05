import os
import re
import sys

def extract_fastsam_time(filepath):
    with open(filepath, "r") as f:
        for line in f:
            match = re.match(r"\s*total:\s*([0-9]*\.?[0-9]+)", line)
            if match:
                return float(match.group(1))
    return None


def compute_average_fastsam_time(directory):
    fastsam_times = []

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            time = extract_fastsam_time(filepath)
            if time is not None:
                fastsam_times.append(time)
            else:
                print(f"[Warning] No total: time found in {filename}")

    if not fastsam_times:
        raise RuntimeError("No total: times found in any file.")

    average = sum(fastsam_times) / len(fastsam_times)
    return average, fastsam_times


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python avg_fastsam_time.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    avg, values = compute_average_fastsam_time(directory)

    print(f"Processed {len(values)} files")
    print(f"Average image processing time: {avg:.4f} s")

