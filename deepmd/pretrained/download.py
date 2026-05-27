# SPDX-License-Identifier: LGPL-3.0-or-later
"""Download and resolve pretrained model files."""

from __future__ import (
    annotations,
)

import concurrent.futures
import hashlib
import logging
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import (
    Path,
)
from typing import (
    Any,
)

from deepmd.pretrained.registry import (
    MODEL_REGISTRY,
)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "deepmd" / "pretrained" / "models"
DOWNLOAD_TIMEOUT_SECONDS = 120
SOURCE_PROBE_TIMEOUT_SECONDS = 8


def _validate_download_url(url: str) -> None:
    """Validate that download URL uses HTTPS scheme."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Unsupported URL scheme for download: {parsed.scheme}")


def _sha256sum(path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _model_download_urls(model_info: dict[str, Any]) -> list[str]:
    """Return candidate download URLs (deduplicated and ordered)."""
    candidates: list[str] = []
    raw_urls = model_info.get("urls")
    if isinstance(raw_urls, list):
        candidates.extend(item for item in raw_urls if isinstance(item, str))

    if not candidates and isinstance(model_info.get("url"), str):
        # backward compatibility
        candidates.append(model_info["url"])

    seen: set[str] = set()
    unique: list[str] = []
    for url in candidates:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def _probe_download_url(url: str) -> float | None:
    """Probe one URL and return latency seconds if reachable; else None."""
    _validate_download_url(url)
    request = urllib.request.Request(
        url,
        headers={"Range": "bytes=0-0"},
        method="GET",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=SOURCE_PROBE_TIMEOUT_SECONDS):
            pass
    except (urllib.error.URLError, OSError, ValueError):
        return None

    return time.monotonic() - start


def _rank_download_urls(urls: list[str]) -> list[str]:
    """Rank candidate URLs by probe latency (fastest first)."""
    if len(urls) <= 1:
        return urls

    results: dict[str, float] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(urls))) as exe:
        future_to_url = {exe.submit(_probe_download_url, url): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            latency = future.result()
            if latency is not None:
                results[url] = latency

    ranked_ok = sorted(results, key=lambda url: results[url])
    ranked_fail = [url for url in urls if url not in results]
    return ranked_ok + ranked_fail


def _download_file(url: str, destination: Path) -> None:
    """Download URL content to destination atomically."""
    _validate_download_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")

    try:
        with (
            urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
            tmp_path.open("wb") as out_file,
        ):
            shutil.copyfileobj(response, out_file)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    tmp_path.replace(destination)


def download_model(
    model_name: str,
    *,
    cache_dir: Path | None = None,
    logger: logging.Logger | None = None,
) -> Path:
    """Download one model and return local path.

    The function will probe all configured sources, try the fastest reachable
    source first, and then fallback to others when failure happens.
    """
    log = logger or logging.getLogger(__name__)

    model_info = MODEL_REGISTRY.get(model_name)
    if model_info is None:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    target_dir = cache_dir or DEFAULT_CACHE_DIR
    output_path = target_dir / str(model_info["filename"])
    expected_sha256 = str(model_info["sha256"])

    if output_path.exists():
        actual = _sha256sum(output_path)
        if actual == expected_sha256:
            log.info("Model '%s' already exists at: %s", model_name, output_path)
            return output_path
        log.warning(
            "Cached file for '%s' failed SHA256 check, re-downloading...",
            model_name,
        )
        output_path.unlink(missing_ok=True)

    urls = _model_download_urls(model_info)
    if not urls:
        raise RuntimeError(f"No download URL configured for model '{model_name}'")

    ranked_urls = _rank_download_urls(urls)
    if len(ranked_urls) > 1:
        log.info(
            "Selecting fastest source among %d candidates...",
            len(ranked_urls),
        )

    for idx, url in enumerate(ranked_urls, start=1):
        log.info(
            "Downloading '%s' (source %d/%d): %s",
            model_name,
            idx,
            len(ranked_urls),
            url,
        )
        try:
            _download_file(url, output_path)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            log.warning("Download attempt failed from %s: %s", url, exc)
            continue

        actual = _sha256sum(output_path)
        if actual != expected_sha256:
            output_path.unlink(missing_ok=True)
            log.warning("SHA256 verification failed from source: %s", url)
            log.warning("Expected: %s", expected_sha256)
            log.warning("Actual:   %s", actual)
            continue

        log.info("Downloaded '%s' to: %s", model_name, output_path)
        return output_path

    raise RuntimeError(f"Failed to download model '{model_name}' from all sources")


def resolve_model_path(
    model_name: str,
    *,
    cache_dir: Path | None = None,
    logger: logging.Logger | None = None,
) -> Path:
    """Resolve model alias to verified local file, downloading if needed."""
    target_dir = cache_dir or DEFAULT_CACHE_DIR
    model_info = MODEL_REGISTRY.get(model_name)
    if model_info is None:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    return download_model(model_name, cache_dir=target_dir, logger=logger)
