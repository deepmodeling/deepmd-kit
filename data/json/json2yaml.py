#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from warnings import warn

import yaml


def _main():
    parser = argparse.ArgumentParser(
        description="convert json config file to yaml",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # get all json files in dir
    jsons = [p for p in Path.cwd().glob("*.json")]
    # use the newest as autosuggestion
    jsons.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    jfile = jsons[0]
    yfile = jfile.with_suffix(".yaml")

    parser.add_argument("INPUT", default=jfile, type=Path, nargs="?",
                        help="input json file")
    parser.add_argument("OUTPUT", default=yfile, type=Path, nargs="?",
                        help="output yaml file")
    args = parser.parse_args()

    with args.INPUT.open("r") as infile, args.OUTPUT.open("w") as outfile:
        yaml.dump(json.load(infile), outfile, default_flow_style=False,
                  sort_keys=False)

    warn("The order of the keys won't be preserved!", SyntaxWarning)
    warn("_comment keys will also be lostt in the conversion")

if __name__ == "__main__":
    _main()
