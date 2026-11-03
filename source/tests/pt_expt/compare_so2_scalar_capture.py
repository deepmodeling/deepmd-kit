# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare same-frame repeats from benchmark_so2_scalar_capture.py.

Run with --torch DIR --base DIR --candidate DIR. An OOM reference is unavailable,
not a pass. Output is JSON; absolute units follow the ASE energy/force/virial/stress
arrays. Relative errors use max(abs(reference), 1e-8) as the denominator.
"""
# ruff: noqa: T201 -- standalone comparison CLI

import argparse
import json
from itertools import (
    product,
)
from pathlib import (
    Path,
)

import numpy as np


def compare(candidate, reference):
    """Compare all 3x3 repeat pairs for each of the three corresponding frames."""
    fields = {}
    for field in ("energy", "forces", "virial", "stress"):
        rows = []
        for frame, i, j in product(range(3), range(3), range(3)):
            x = candidate[f"{i}_{frame}_{field}"]
            y = reference[f"{j}_{frame}_{field}"]
            assert x.shape == y.shape
            delta = np.abs(x - y)
            rows.append(
                {
                    "max_abs": float(delta.max()),
                    "mean_abs": float(delta.mean()),
                    "max_relative": float((delta / np.maximum(abs(y), 1e-8)).max()),
                    "max_ratio": float((delta / (1e-5 + 1e-4 * abs(y))).max()),
                    "finite": bool(np.isfinite(x).all() and np.isfinite(y).all()),
                }
            )
        fields[field] = {
            k: max(row[k] for row in rows)
            for k in ("max_abs", "mean_abs", "max_relative", "max_ratio")
        }
        fields[field]["passed"] = all(
            row["finite"] and row["max_ratio"] <= 1 for row in rows
        )
    return fields


def validate_identity(result, reference, manifest, reference_manifest):
    """Reject cross-checkpoint or cross-input comparisons before scoring."""
    for key, left, right in (
        ("input_sha256", result, reference),
        ("model_sha256", manifest, reference_manifest),
    ):
        if not left.get(key) or left[key] != right.get(key):
            raise ValueError(f"Comparison requires matching {key}")


def main():
    """Load benchmark routes and report compatible repeated comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    for route in ("torch", "base", "candidate"):
        parser.add_argument(f"--{route}", type=Path, required=True)
    args = parser.parse_args()
    results, arrays, out, manifests = {}, {}, {}, {}
    for route, path in vars(args).items():
        results[route] = json.loads((path / "result.json").read_text())
        out[route] = {"status": results[route]["status"]}
        if results[route]["status"] == "completed":
            manifests[route] = json.loads((path / "manifest.json").read_text())
            arrays[route] = dict(np.load(path / "outputs.npz"))
            out[route]["repeat"] = compare(arrays[route], arrays[route])
    for route in arrays:
        for reference in ("torch", "base"):
            if route != reference and reference in arrays:
                validate_identity(
                    results[route],
                    results[reference],
                    manifests[route],
                    manifests[reference],
                )
                out[route][f"vs_{reference}"] = compare(
                    arrays[route], arrays[reference]
                )
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
