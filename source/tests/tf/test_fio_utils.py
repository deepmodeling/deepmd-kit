# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.tf.nvnmd.utils.fio import (
    Fio,
    FioTxt,
)


def test_get_file_list(tmp_path):
    """get_file_list should handle non-existent paths and collect files recursively."""
    # create directory with one file
    subdir = tmp_path / "sub"
    subdir.mkdir()
    file_path = subdir / "file.txt"
    file_path.write_text("hello")

    fio = Fio()

    # existing directory returns the file
    files = fio.get_file_list(str(tmp_path))
    assert files == [str(file_path)]

    # non-existent directory should return an empty list
    missing = tmp_path / "missing"
    assert fio.get_file_list(str(missing)) == []


def test_fiotxt_load_default(tmp_path):
    """FioTxt.load should return default empty list when file does not exist."""
    fio_txt = FioTxt()
    missing = tmp_path / "no.txt"
    assert fio_txt.load(str(missing)) == []
