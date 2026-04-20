# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import (
    annotations,
)

import os
from pathlib import (
    Path,
)

import model_convert


def _set_mtime(path: Path, mtime_ns: int) -> None:
    os.utime(path, ns=(mtime_ns, mtime_ns))


def test_skips_up_to_date_output(tmp_path, monkeypatch) -> None:
    source = tmp_path / "model.pbtxt"
    output = tmp_path / "model.pb"
    source.write_text("source", encoding="utf-8")
    output.write_text("converted", encoding="utf-8")

    base_mtime_ns = 1_700_000_000_000_000_000
    _set_mtime(source, base_mtime_ns)
    _set_mtime(output, base_mtime_ns)

    called = False

    def fake_run(*args, **kwargs) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(model_convert.sp, "run", fake_run)

    assert model_convert.ensure_converted_pb(source, output) == output.resolve()
    assert output.read_text(encoding="utf-8") == "converted"
    assert not called


def test_rebuilds_stale_output(tmp_path, monkeypatch) -> None:
    source = tmp_path / "model.pbtxt"
    output = tmp_path / "model.pb"
    source.write_text("new", encoding="utf-8")
    output.write_text("old", encoding="utf-8")

    base_mtime_ns = 1_700_000_000_000_000_000
    _set_mtime(output, base_mtime_ns)
    _set_mtime(source, base_mtime_ns + 1_000_000_000)

    tmp_output: Path | None = None

    def fake_run(cmd: list[str], check: bool) -> None:
        nonlocal tmp_output
        assert check is True
        tmp_output = Path(cmd[-1])
        tmp_output.write_text("converted", encoding="utf-8")

    monkeypatch.setattr(model_convert.sp, "run", fake_run)

    assert model_convert.ensure_converted_pb(source, output) == output.resolve()
    assert output.read_text(encoding="utf-8") == "converted"
    assert tmp_output is not None
    assert not tmp_output.exists()


def test_breaks_stale_lock_before_converting(tmp_path, monkeypatch) -> None:
    source = tmp_path / "model.pbtxt"
    output = tmp_path / "model.pb"
    lock_file = tmp_path / ".model.pb.lock"
    source.write_text("source", encoding="utf-8")
    lock_file.write_text("pid=999999\n", encoding="utf-8")

    def fake_run(cmd: list[str], check: bool) -> None:
        assert check is True
        assert lock_file.exists()
        assert lock_file.read_text(encoding="utf-8") == f"pid={os.getpid()}\n"
        Path(cmd[-1]).write_text("converted", encoding="utf-8")

    monkeypatch.setattr(model_convert.sp, "run", fake_run)

    assert model_convert.ensure_converted_pb(source, output) == output.resolve()
    assert output.read_text(encoding="utf-8") == "converted"
    assert not lock_file.exists()
