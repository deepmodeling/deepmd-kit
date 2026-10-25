"""Run the disclosed three-arm manual CUDA matrix with explicit source/model paths."""

# ruff: noqa: T201 -- manual benchmark
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

P = argparse.ArgumentParser(description=__doc__)
P.add_argument("--base-root", type=Path, required=True)
P.add_argument("--candidate-root", type=Path, required=True)
P.add_argument("--model-map", type=Path, required=True)
P.add_argument("--out-root", type=Path, required=True)
P.add_argument("--gpu", type=int, required=True)
P.add_argument(
    "--models", nargs="+", choices=["Neo", "Nano", "Mini", "Pro"], required=True
)
P.add_argument(
    "--sizes", type=int, nargs="+", choices=[96, 512, 4096], default=[96, 512, 4096]
)
P.add_argument("--repeats", type=int, default=3)
P.add_argument("--skip-existing", action="store_true")
a = P.parse_args()
models = json.loads(a.model_map.read_text())
assert a.repeats >= 3
for root in (a.base_root, a.candidate_root):
    assert (root / "deepmd").is_dir(), root
records = []
for model in a.models:
    checkpoint = Path(models[model]).resolve()
    assert checkpoint.is_file(), checkpoint
    for system in ["diamond", "bcccu"] if model == "Neo" else ["diamond"]:
        dims = (
            {96: (2, 2, 3), 512: (4, 4, 4), 4096: (8, 8, 8)}
            if system == "diamond"
            else {96: (4, 4, 3), 512: (8, 8, 4), 4096: (16, 16, 8)}
        )
        label = model if system == "diamond" else model + "Cu"
        for n in a.sizes:
            for repeat in range(a.repeats):
                for mode in (
                    ["TORCH", "BASE", "ECS"]
                    if repeat % 2 == 0
                    else ["ECS", "BASE", "TORCH"]
                ):
                    out = (
                        a.out_root.resolve()
                        / "results"
                        / f"{label}_n{n}_{mode}_p{repeat}"
                    )
                    if (out / "result.json").exists():
                        if a.skip_existing:
                            continue
                        raise FileExistsError(
                            f"Use a fresh output root; not overwriting {out}"
                        )
                    busy = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "-i",
                            str(a.gpu),
                            "--query-compute-apps=pid",
                            "--format=csv,noheader",
                        ],
                        text=True,
                    ).strip()
                    if busy:
                        raise RuntimeError(
                            f"GPU busy; do not kill other processes: {busy}"
                        )
                    out.parent.mkdir(parents=True, exist_ok=True)
                    command = [
                        sys.executable,
                        str(Path(__file__).with_name("worker.py")),
                        "--root",
                        str(
                            (
                                a.candidate_root if mode == "ECS" else a.base_root
                            ).resolve()
                        ),
                        "--model",
                        str(checkpoint),
                        "--out",
                        str(out),
                        "--cache",
                        str(a.out_root.resolve() / "cache" / f"{label}_{mode}"),
                        "--gpu",
                        str(a.gpu),
                        "--mode",
                        mode,
                        "--nrep",
                        "2",
                        "--repeat-cell",
                        *map(str, dims[n]),
                        "--system",
                        system,
                        "--warmup",
                        "8",
                        "--samples",
                        "30",
                    ]
                    print("START", out.name, flush=True)
                    with out.with_suffix(".log").open("w") as stream:
                        proc = subprocess.run(
                            command,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONHASHSEED=str(repeat % 2)),
                            check=False,
                        )
                    records.append(
                        {"run": out.name, "exit": proc.returncode, "command": command}
                    )
                    (a.out_root / f"commands_gpu{a.gpu}.json").write_text(
                        json.dumps(records, indent=2)
                    )
                    print("DONE", out.name, proc.returncode, flush=True)
