import csv
import numpy as np
from typing import Sequence

def feed_csv(file):
    reader = csv.DictReader(file, delimiter=',')
    header = reader.fieldnames

    if not isinstance(header, Sequence):
        print("No header supplied.")
        exit(0)

    result = { key: [] for key in header }
    for row in reader:
        for key in header:
            result[key].append(row[key])

    result_np = { key: np.empty(0, dtype=np.float64) for key in header }
    for key in header:
        result_np[key] = np.array(result[key], dtype=np.float64)

    return result_np
