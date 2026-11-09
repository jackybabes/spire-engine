"""Checks for train_fights.append_csv (plain asserts, no pytest needed):

    python python/tests/test_csv.py

A new file gets the fields as its header; an existing one keeps its columns and rows, and gains
the fields it lacks (a resumed run logging what newer code logs)."""
import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

from training.train_fights import append_csv


def main():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "log.csv")
        append_csv(path, ["update", "won"], [{"update": 1, "won": 0.5}])
        append_csv(path, ["update", "won"], [{"update": 2, "won": 0.25, "ignored": 1}])
        append_csv(path, ["update", "hp_lost", "won"], [{"update": 3, "won": 0.75, "hp_lost": 12.5}])
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        with open(path, newline="") as f:
            head = next(csv.reader(f))
    assert head == ["update", "won", "hp_lost"], head
    assert [(r["update"], r["won"], r["hp_lost"]) for r in rows] == [("1", "0.5", ""), ("2", "0.25", ""), ("3", "0.75", "12.5")], rows
    print("csv: ok")


if __name__ == "__main__":
    main()
