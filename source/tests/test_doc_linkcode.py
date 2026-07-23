# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import (
    Path,
)
from types import (
    SimpleNamespace,
)

from doc import (
    github_linkcode,
)


class AutoapiObject:
    """Minimal AutoAPI object used to exercise source-link resolution."""

    def __init__(self, object_id: str, **metadata) -> None:
        self.id = object_id
        self.obj = metadata


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
    commit = "1f94e04b7f596c309b7efab4e7630ed78e85a1f1"

    monkeypatch.setattr(github_linkcode, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setenv("READTHEDOCS_GIT_COMMIT_HASH", commit)
    github_linkcode.get_git_commit.cache_clear()
    github_linkcode.collect_autoapi_source_locations(app)

    assert github_linkcode.linkcode_resolve(
        "py",
        {"module": "deepmd.api", "fullname": "public_function"},
    ) == (
        "https://github.com/deepmodeling/deepmd-kit/blob/"
        f"{commit}/deepmd/implementation.py#L12-L24"
    )


def test_linkcode_rejects_non_python_domains(monkeypatch) -> None:
    monkeypatch.setenv(
        "READTHEDOCS_GIT_COMMIT_HASH",
        "1f94e04b7f596c309b7efab4e7630ed78e85a1f1",
    )
    github_linkcode.get_git_commit.cache_clear()

    assert (
        github_linkcode.linkcode_resolve(
            "cpp", {"module": "deepmd.api", "fullname": "public_function"}
        )
        is None
    )
