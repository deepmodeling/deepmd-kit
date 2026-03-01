# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for pretrained download/resolve helpers."""

from __future__ import (
    annotations,
)

import hashlib
import tempfile
import unittest
import urllib.error
from pathlib import (
    Path,
)
from unittest.mock import (
    patch,
)

from deepmd.pretrained import download as dl


class TestPretrainedDownload(unittest.TestCase):
    """Test download helper behavior."""

    def test_model_download_urls_prefers_urls(self) -> None:
        info = {
            "urls": ["https://a", "https://a", "https://b"],
            "url": "https://legacy",
        }
        self.assertEqual(dl._model_download_urls(info), ["https://a", "https://b"])

    def test_rank_download_urls(self) -> None:
        with patch.object(
            dl,
            "_probe_download_url",
            side_effect=lambda url: {
                "https://a": 0.3,
                "https://b": 0.1,
                "https://c": None,
            }[url],
        ):
            ranked = dl._rank_download_urls(["https://a", "https://b", "https://c"])

        self.assertEqual(ranked, ["https://b", "https://a", "https://c"])

    def test_download_model_fallback_on_failure(self) -> None:
        payload = b"payload"
        expected = hashlib.sha256(payload).hexdigest()
        model_name = "DPA-3.2-5M"

        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)

            with patch.object(
                dl,
                "MODEL_REGISTRY",
                {
                    model_name: {
                        "filename": "model.pt",
                        "sha256": expected,
                        "urls": ["https://a", "https://b"],
                    }
                },
            ):
                with patch.object(
                    dl,
                    "_rank_download_urls",
                    return_value=["https://a", "https://b"],
                ):

                    def fake_download(url: str, destination: Path) -> None:
                        if url == "https://a":
                            raise urllib.error.URLError("timeout")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(payload)

                    with patch.object(dl, "_download_file", side_effect=fake_download):
                        path = dl.download_model(model_name, cache_dir=cache_dir)

            self.assertTrue(path.exists())
            self.assertEqual(path.read_bytes(), payload)

    def test_resolve_model_path_cached(self) -> None:
        payload = b"payload"
        expected = hashlib.sha256(payload).hexdigest()
        model_name = "DPA-3.2-5M"

        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            target = cache_dir / "model.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

            with patch.object(
                dl,
                "MODEL_REGISTRY",
                {
                    model_name: {
                        "filename": "model.pt",
                        "sha256": expected,
                        "urls": ["https://a"],
                    }
                },
            ):
                with patch.object(dl, "_download_file") as mocked_download:
                    path = dl.resolve_model_path(model_name, cache_dir=cache_dir)

            self.assertEqual(path, target)
            mocked_download.assert_not_called()
