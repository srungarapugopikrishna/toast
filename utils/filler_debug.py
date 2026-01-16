import csv
import os


def write_fillers_csv(out_dir, filler_rows):
    """
    Writes detected fillers to fillers.csv for debugging.
    """
    if not filler_rows:
        print("ℹ️ No fillers detected (fillers.csv not written)")
        return

    path = os.path.join(out_dir, "fillers.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "start", "end", "type"])
        for r in filler_rows:
            writer.writerow(r)

    print(f"🧪 Filler debug CSV written → {path}")
