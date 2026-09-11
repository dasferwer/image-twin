"""Считаем найденные копии и ложные совпадения на двух непересекающихся наборах снимков."""

import hashlib
import json
import time
from pathlib import Path

from fixtures import variants

from imagetwin.vision import compare, encoder, normalize

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.monotonic()
    sources = json.loads((ROOT / "data/source.json").read_text())
    records = {}
    for source in sources:
        path = ROOT / "data" / source["file"]
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == source["sha256"]
        records[path.stem] = {
            "original": encoder().extract(normalize(data)[0]),
            "variants": {
                name: encoder().extract(normalize(value)[0])
                for name, value in variants(data).items()
            },
        }
    splits = {
        "calibration": ["camera", "coffee", "horse", "microaneurysms"],
        "evaluation": ["chelsea", "eagle", "human_mitosis", "astronaut"],
    }
    report = {
        "encoder_version": encoder().version,
        "thresholds": {
            "phash_max_distance": 8,
            "minimum_cosine": 0.92,
            "minimum_inliers": 12,
            "minimum_inlier_ratio": 0.5,
            "minimum_matched_area": 0.06,
        },
        "scope": "8 public reference images; transformed copies, not a marketplace production benchmark",
        "splits": {},
    }
    for split, names in splits.items():
        counters = {
            method: {
                "true_positives": 0,
                "false_negatives": 0,
                "false_positives": 0,
                "true_negatives": 0,
            }
            for method in ["phash_only", "combined"]
        }
        outcomes = []
        for name in names:
            for transformation, query in records[name]["variants"].items():
                for candidate in names:
                    result = compare(query, records[candidate]["original"])
                    expected = name == candidate
                    for method, predicted in [
                        ("phash_only", result["phash_distance"] <= 8),
                        ("combined", result["duplicate"]),
                    ]:
                        key = (
                            "true_positives"
                            if predicted and expected
                            else "false_positives"
                            if predicted
                            else "false_negatives"
                            if expected
                            else "true_negatives"
                        )
                        counters[method][key] += 1
                    if expected:
                        outcomes.append(
                            {"source": name, "transformation": transformation, **result}
                        )
        for count in counters.values():
            tp, fn, fp = count["true_positives"], count["false_negatives"], count["false_positives"]
            count["recall"] = tp / (tp + fn)
            count["precision"] = tp / (tp + fp) if tp + fp else 0
        report["splits"][split] = {
            "source_images": names,
            "metrics": counters,
            "positive_pairs": outcomes,
        }
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    (ROOT / "docs/evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({name: values["metrics"] for name, values in report["splits"].items()}, indent=2)
    )


if __name__ == "__main__":
    main()
