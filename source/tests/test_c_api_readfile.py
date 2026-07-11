# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression test for ``DP_ReadFileToChar2`` / ``read_file_to_string``.

See https://github.com/deepmodeling/deepmd-kit/issues/5620 for context.
Previously ``DP_ReadFileToChar2`` reported the original file size but
returned a buffer produced by ``string_to_char``, which trims trailing
whitespace before allocating/copying.  The C++ wrapper then reconstructed a
``std::string`` with the reported (larger) size, causing an over-read of the
shorter allocation.  This test verifies that exact bytes are preserved,
including trailing whitespace.
"""

import ctypes
import pathlib

import pytest


def _load_c_lib():
    """Load the DeePMD C library (``libdeepmd_c.so``).

    The library ships with the Python package; we search the standard
    locations and fall back to the build tree.
    """
    import deepmd

    candidates = [
        pathlib.Path(deepmd.__file__).parent / "lib" / "libdeepmd_c.so",
        pathlib.Path(__file__).resolve().parents[2] / "dp" / "lib" / "libdeepmd_c.so",
    ]
    for path in candidates:
        if path.exists():
            return ctypes.CDLL(str(path))
    pytest.skip("libdeepmd_c.so not found")
    return None


@pytest.fixture(scope="module")
def c_lib():
    """Module-scoped fixture that loads the C library and configures
    the DP_ReadFileToChar2 / DP_DeleteChar signatures once.
    """
    lib = _load_c_lib()
    lib.DP_ReadFileToChar2.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.DP_ReadFileToChar2.restype = ctypes.c_void_p
    lib.DP_DeleteChar.argtypes = [ctypes.c_void_p]
    lib.DP_DeleteChar.restype = None
    return lib


def test_readfiletochar2_preserves_trailing_whitespace(tmp_path, c_lib):
    """Files ending in whitespace must return exact size and bytes."""
    # Write a temporary file whose content ends with " \n" (trailing
    # whitespace).  This is the exact scenario that was broken: the old
    # code trimmed the whitespace but reported the original size.
    content = b"hello world \n"  # ends with space + newline
    tmp_file = tmp_path / "test_exact_bytes.txt"
    tmp_file.write_bytes(content)

    # Call DP_ReadFileToChar2 ----------------------------------------------
    # Use c_void_p for the return type so ctypes does not truncate the
    # buffer at the first null byte (c_char_p would do that).
    size = ctypes.c_int(0)
    c_buf = c_lib.DP_ReadFileToChar2(str(tmp_file).encode("utf-8"), ctypes.byref(size))
    assert c_buf is not None, "DP_ReadFileToChar2 returned NULL"

    size_val = size.value
    # Negative size indicates an error (the error message is in the buffer).
    assert size_val >= 0, (
        f"DP_ReadFileToChar2 error: {ctypes.string_at(c_buf, -size_val)}"
    )

    # The reported size must match the file size exactly.
    assert size_val == len(content), (
        f"Size mismatch: file has {len(content)} bytes but "
        f"DP_ReadFileToChar2 reported {size_val}"
    )

    # The returned buffer must contain the exact file bytes, including
    # trailing whitespace.
    returned_bytes = ctypes.string_at(c_buf, size_val)
    assert returned_bytes == content, (
        f"Content mismatch: expected {content!r} but got {returned_bytes!r}"
    )

    # Clean up the allocated buffer.
    c_lib.DP_DeleteChar(c_buf)


def test_readfiletochar2_preserves_no_trailing_whitespace(tmp_path, c_lib):
    """Sanity check: files without trailing whitespace still work."""
    content = b"hello world"
    tmp_file = tmp_path / "test_no_trailing.txt"
    tmp_file.write_bytes(content)

    size = ctypes.c_int(0)
    c_buf = c_lib.DP_ReadFileToChar2(str(tmp_file).encode("utf-8"), ctypes.byref(size))
    assert size.value == len(content)
    assert ctypes.string_at(c_buf, size.value) == content
    c_lib.DP_DeleteChar(c_buf)
