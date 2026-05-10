import csv
import os
from pathlib import Path


RAW_DIR = Path("data/raw")
CSV_DIR = Path("data/csv")


def convert_dat_to_csv(dat_path: Path, csv_path: Path) -> int:
    """Convert a FIMI .dat transaction file to a two-column CSV."""
    rows = 0
    with dat_path.open("r", encoding="utf-8") as src, csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        writer = csv.writer(dst)
        writer.writerow(["transaction_id", "items"])
        for transaction_id, line in enumerate(src):
            items = " ".join(line.strip().split())
            if items:
                writer.writerow([transaction_id, items])
                rows += 1
    return rows


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_DIR.exists():
        print("No data/raw directory found.")
        return

    converted = []
    for dat_path in sorted(RAW_DIR.glob("*.dat")):
        csv_path = CSV_DIR / f"{dat_path.stem}.csv"
        rows = convert_dat_to_csv(dat_path, csv_path)
        converted.append((dat_path.name, csv_path.as_posix(), rows))

    if not converted:
        print("No .dat files found in data/raw.")
        return

    for source, target, rows in converted:
        print(f"{source} -> {target} ({rows} transactions)")


if __name__ == "__main__":
    main()
