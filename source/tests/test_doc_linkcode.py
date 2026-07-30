# SPDX-License-Identifier: LGPL-3.0-or-later
import subprocess
from pathlib import (
    Path,
)
from types import (
    SimpleNamespace,
)

import pytest

from doc import (
    github_linkcode,
)

COMMIT = "1f94e04b7f596c309b7efab4e7630ed78e85a1f1"


class AutoapiObject:
    """Minimal AutoAPI object used to exercise source-link resolution."""

    def __init__(self, object_id: str, **metadata) -> None:
        self.id = object_id
        self.obj = metadata


@pytest.fixture(autouse=True)
def reset_linkcode_state():
    """Keep module caches isolated between source-link tests."""
    github_linkcode._source_locations.clear()
    github_linkcode._unresolved_source_objects.clear()
    github_linkcode.get_git_commit.cache_clear()
    yield
    github_linkcode._source_locations.clear()
    github_linkcode._unresolved_source_objects.clear()
    clear_cache = getattr(github_linkcode.get_git_commit, "cache_clear", None)
    if clear_cache is not None:
        clear_cache()


def test_linkcode_uses_original_object_and_rtd_commit(
    monkeypatch, tmp_path: Path
) -> None:
    repository_root = tmp_path.resolve()
    module_path = repository_root / "deepmd" / "implementation.py"
    module_path.parent.mkdir()
    module_path.touch()

    objects = {
        "deepmd.api": AutoapiObject(
            "deepmd.api", file_path=str(repository_root / "deepmd" / "api.py")
        ),
        "deepmd.api.public_function": AutoapiObject(
            "deepmd.api.public_function",
            original_path="deepmd.implementation.public_function",
        ),
        "deepmd.implementation": AutoapiObject(
            "deepmd.implementation", file_path=str(module_path)
        ),
        "deepmd.implementation.public_function": AutoapiObject(
            "deepmd.implementation.public_function",
            from_line_no=12,
            to_line_no=24,
        ),
    }
    app = SimpleNamespace(
        env=SimpleNamespace(autoapi_all_objects=objects),
    )
    monkeypatch.setattr(github_linkcode, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setenv("READTHEDOCS_GIT_COMMIT_HASH", COMMIT)
    github_linkcode.collect_autoapi_source_locations(app)

    assert github_linkcode.linkcode_resolve(
        "py",
        {"module": "deepmd.api", "fullname": "public_function"},
    ) == (
        "https://github.com/deepmodeling/deepmd-kit/blob/"
        f"{COMMIT}/deepmd/implementation.py#L12-L24"
    )


def test_linkcode_rejects_non_python_domains(monkeypatch) -> None:
    monkeypatch.setattr(
        github_linkcode,
        "get_git_commit",
        lambda: pytest.fail("domain rejection must not inspect Git state"),
    )

    assert (
        github_linkcode.linkcode_resolve(
            "cpp", {"module": "deepmd.api", "fullname": "public_function"}
        )
        is None
    )


def test_reexported_class_member_uses_the_defining_module(
    monkeypatch, tmp_path: Path
) -> None:
    """Members inherit a re-exported class's original source namespace."""
    repository_root = tmp_path.resolve()
    public_module = repository_root / "deepmd" / "__init__.py"
    implementation_module = repository_root / "deepmd" / "implementation.py"
    public_module.parent.mkdir()
    public_module.touch()
    implementation_module.touch()
    objects = {
        "deepmd": AutoapiObject("deepmd", file_path=str(public_module)),
        "deepmd.Widget": AutoapiObject(
            "deepmd.Widget",
            original_path="deepmd.implementation.Widget",
        ),
        "deepmd.Widget.spin": AutoapiObject("deepmd.Widget.spin"),
        "deepmd.implementation": AutoapiObject(
            "deepmd.implementation",
            file_path=str(implementation_module),
        ),
        "deepmd.implementation.Widget": AutoapiObject(
            "deepmd.implementation.Widget",
            from_line_no=4,
            to_line_no=17,
        ),
        "deepmd.implementation.Widget.spin": AutoapiObject(
            "deepmd.implementation.Widget.spin",
            from_line_no=8,
            to_line_no=10,
        ),
    }

    monkeypatch.setattr(github_linkcode, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setenv("READTHEDOCS_GIT_COMMIT_HASH", COMMIT)
    github_linkcode.collect_autoapi_source_locations(
        SimpleNamespace(env=SimpleNamespace(autoapi_all_objects=objects))
    )

    assert github_linkcode.linkcode_resolve(
        "py", {"module": "deepmd", "fullname": "Widget.spin"}
    ) == (
        "https://github.com/deepmodeling/deepmd-kit/blob/"
        f"{COMMIT}/deepmd/implementation.py#L8-L10"
    )


def test_unresolved_member_does_not_fall_back_to_importing_module(
    monkeypatch, tmp_path: Path
) -> None:
    """Known line-less members produce no link instead of a wrong file link."""
    repository_root = tmp_path.resolve()
    module_path = repository_root / "deepmd" / "__init__.py"
    module_path.parent.mkdir()
    module_path.touch()
    objects = {
        "deepmd": AutoapiObject("deepmd", file_path=str(module_path)),
        "deepmd.Widget": AutoapiObject("deepmd.Widget", from_line_no=4, to_line_no=9),
        "deepmd.Widget.ATTR": AutoapiObject("deepmd.Widget.ATTR"),
    }

    monkeypatch.setattr(github_linkcode, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setenv("READTHEDOCS_GIT_COMMIT_HASH", COMMIT)
    github_linkcode.collect_autoapi_source_locations(
        SimpleNamespace(env=SimpleNamespace(autoapi_all_objects=objects))
    )

    assert (
        github_linkcode.linkcode_resolve(
            "py", {"module": "deepmd", "fullname": "Widget.ATTR"}
        )
        is None
    )


def test_collect_rejects_missing_or_empty_autoapi_metadata() -> None:
    """AutoAPI API drift must fail visibly instead of deleting all links."""
    with pytest.raises(RuntimeError, match="metadata is unavailable"):
        github_linkcode.collect_autoapi_source_locations(
            SimpleNamespace(env=SimpleNamespace())
        )
    with pytest.raises(RuntimeError, match="empty object graph"):
        github_linkcode.collect_autoapi_source_locations(
            SimpleNamespace(env=SimpleNamespace(autoapi_all_objects={}))
        )
    with pytest.raises(RuntimeError, match="no module source paths"):
        github_linkcode.collect_autoapi_source_locations(
            SimpleNamespace(
                env=SimpleNamespace(
                    autoapi_all_objects={"deepmd.value": AutoapiObject("deepmd.value")}
                )
            )
        )


def test_collect_rejects_sources_outside_the_repository(
    monkeypatch, tmp_path: Path
) -> None:
    """Imported external objects must never receive repository URLs."""
    repository_root = (tmp_path / "repository").resolve()
    repository_root.mkdir()
    external_module = (tmp_path / "external.py").resolve()
    external_module.touch()
    objects = {
        "external": AutoapiObject("external", file_path=str(external_module)),
        "external.value": AutoapiObject("external.value", from_line_no=1, to_line_no=1),
    }

    monkeypatch.setattr(github_linkcode, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setenv("READTHEDOCS_GIT_COMMIT_HASH", COMMIT)
    github_linkcode.collect_autoapi_source_locations(
        SimpleNamespace(env=SimpleNamespace(autoapi_all_objects=objects))
    )

    assert (
        github_linkcode.linkcode_resolve(
            "py", {"module": "external", "fullname": "value"}
        )
        is None
    )


def test_original_resolution_stops_on_cycles_and_dangling_paths() -> None:
    """Malformed re-export metadata cannot loop or discard the last safe object."""
    cycle_a = AutoapiObject("pkg.a", original_path="pkg.b")
    cycle_b = AutoapiObject("pkg.b", original_path="pkg.a")
    dangling = AutoapiObject("pkg.dangling", original_path="pkg.missing")
    objects = {"pkg.a": cycle_a, "pkg.b": cycle_b, "pkg.dangling": dangling}

    assert (
        github_linkcode._resolve_original_object("pkg.a", cycle_a, objects) is cycle_a
    )
    assert (
        github_linkcode._resolve_original_object("pkg.dangling", dangling, objects)
        is dangling
    )


def test_get_git_commit_falls_back_to_the_checkout(monkeypatch) -> None:
    """Local documentation builds use git rev-parse when RTD metadata is absent."""
    monkeypatch.delenv("READTHEDOCS_GIT_COMMIT_HASH", raising=False)
    monkeypatch.setattr(
        github_linkcode.subprocess,
        "check_output",
        lambda *args, **kwargs: f"{COMMIT.upper()}\n",
    )

    assert github_linkcode.get_git_commit() == COMMIT


@pytest.mark.parametrize(
    "error",
    [OSError("git missing"), subprocess.CalledProcessError(1, ["git"])],
)
def test_get_git_commit_reports_checkout_failures(monkeypatch, caplog, error) -> None:
    """A checkout without usable Git metadata emits an actionable warning."""
    monkeypatch.delenv("READTHEDOCS_GIT_COMMIT_HASH", raising=False)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(github_linkcode.subprocess, "check_output", fail)

    assert github_linkcode.get_git_commit() is None
    assert "Cannot determine a Git commit" in caplog.text


def test_get_git_commit_rejects_malformed_hashes(monkeypatch, caplog) -> None:
    """Only immutable full hashes are accepted in generated source URLs."""
    monkeypatch.setenv("READTHEDOCS_GIT_COMMIT_HASH", "main")

    assert github_linkcode.get_git_commit() is None
    assert "Ignoring invalid documentation Git commit" in caplog.text


@pytest.mark.parametrize(
    ("location", "expected_suffix"),
    [
        (
            github_linkcode.SourceLocation("deepmd/a b.py", 5, None),
            "deepmd/a%20b.py#L5",
        ),
        (github_linkcode.SourceLocation("deepmd/a.py", 5, 5), "deepmd/a.py#L5"),
        (github_linkcode.SourceLocation("deepmd/a.py", 5, 9), "deepmd/a.py#L5-L9"),
    ],
)
def test_linkcode_formats_line_ranges(monkeypatch, location, expected_suffix) -> None:
    """Single-line, open-ended, and ranged locations form stable anchors."""
    monkeypatch.setenv("READTHEDOCS_GIT_COMMIT_HASH", COMMIT)
    github_linkcode._source_locations["deepmd.api.value"] = location

    result = github_linkcode.linkcode_resolve(
        "py", {"module": "deepmd.api", "fullname": "deepmd.api.value"}
    )

    assert result == (
        f"https://github.com/deepmodeling/deepmd-kit/blob/{COMMIT}/{expected_suffix}"
    )


def test_linkcode_uses_module_fallback_and_requires_a_module(monkeypatch) -> None:
    """Unknown members fall back to their module, but empty module names do not."""
    monkeypatch.setenv("READTHEDOCS_GIT_COMMIT_HASH", COMMIT)
    github_linkcode._source_locations["deepmd.api"] = github_linkcode.SourceLocation(
        "deepmd/api.py", None, None
    )

    assert github_linkcode.linkcode_resolve(
        "py", {"module": "deepmd.api", "fullname": "missing"}
    ) == (f"https://github.com/deepmodeling/deepmd-kit/blob/{COMMIT}/deepmd/api.py")
    assert (
        github_linkcode.linkcode_resolve("py", {"module": "", "fullname": "missing"})
        is None
    )
