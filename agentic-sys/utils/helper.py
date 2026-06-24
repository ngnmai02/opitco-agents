# extra helper functions
import csv
import json
from pathlib import Path


def read_csv_from_data(file_name, data_dir=None, delimiter=";"):
    base_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent / "data"
    csv_path = base_dir / file_name

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        data = list(csv.DictReader(csv_file, delimiter=delimiter))

    return json.dump(data, indent=2)

