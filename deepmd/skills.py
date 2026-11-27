# SPDX-License-Identifier: LGPL-3.0-or-later
"""Read the public workflow guides shipped with this version of DeePMD-kit."""

import argparse
import sys
from pathlib import (
    Path,
)


def skill_root() -> Path:
    """Locate installed resources or the adjacent development checkout."""
    package = Path(__file__).resolve().parent
    bundled = package / "_skills"
    if bundled.is_dir():
        return bundled
    checkout = package.parent
    if (checkout / "pyproject.toml").is_file() and (checkout / "skills").is_dir():
        return checkout / "skills"
    raise ValueError("Bundled skills are missing; reinstall the package.")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the offline skills commands on the application parser."""
    parser = subparsers.add_parser(
        "skills", help="Read bundled workflow guides offline."
    )
    actions = parser.add_subparsers(dest="skills_action", required=True)
    actions.add_parser("list", help="List available workflow guides.")
    read = actions.add_parser(
        "read", help="Read a skill or one of its reference files."
    )
    read.add_argument("name", help="Skill name from 'dp skills list'.")
    read.add_argument(
        "path", nargs="?", default="SKILL.md", help="Relative reference path."
    )


def read_skill(name: str, path: str = "SKILL.md") -> str:
    """Read a public text resource without executing it or following escaping paths."""
    root = skill_root().resolve()
    if not name or name in (".", "..") or any(c in name for c in "/\\\0"):
        raise ValueError("invalid skill name")
    parts = path.split("/")
    if (
        not path
        or any(c in path for c in "\\\0:")
        or any(p in ("", ".", "..") for p in parts)
    ):
        raise ValueError("reference path must stay within the skill")
    skill = (root / name).resolve()
    if skill.parent != root or not (skill / "SKILL.md").is_file():
        raise ValueError(f"unknown skill: {name}")
    target = skill.joinpath(*parts).resolve()
    if not target.is_relative_to(skill) or not target.is_file():
        raise ValueError(f"unknown skill reference: {path}")
    try:
        return target.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("skill reference is not UTF-8 text") from exc


def run(args: argparse.Namespace) -> None:
    """Write the requested guide to stdout; report invalid references as CLI errors."""
    try:
        if args.skills_action == "list":
            root = skill_root().resolve()
            for path in sorted(root.glob("*/SKILL.md")):
                if path.parent.resolve().parent == root:
                    sys.stdout.write(path.parent.name + "\n")
            return
        content = read_skill(args.name, args.path)
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"dp skills: {exc}\n")
        raise SystemExit(2) from exc
    sys.stdout.write(content)


def workflow_help(name: str) -> str:
    """Point help readers to an available, version-matched workflow guide."""
    try:
        available = skill_root().joinpath(name, "SKILL.md").is_file()
    except (ValueError, OSError):
        return ""
    if not available:
        return ""
    return f"Workflow guide (read first): dp skills read {name}"
