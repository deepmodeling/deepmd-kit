# SPDX-License-Identifier: LGPL-3.0-or-later
"""Registry of built-in pretrained model sources."""

from typing import (
    Any,
)

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "DPA-3.2-5M": {
        "urls": [
            "https://huggingface.co/deepmodelingcommunity/DPA-3.2-5M/resolve/main/DPA-3.2-5M.pt?download=true",
            "https://hf-mirror.com/deepmodelingcommunity/DPA-3.2-5M/resolve/main/DPA-3.2-5M.pt?download=true",
        ],
        "filename": "DPA-3.2-5M.pt",
        "sha256": "876354744aeaae17b2639a6a690514470273784f2b4836280850f50cbb799165",
    },
    "DPA-3.1-3M": {
        "urls": [
            "https://huggingface.co/deepmodelingcommunity/DPA-3.1-3M/resolve/main/DPA-3.1-3M.pt?download=true",
            "https://hf-mirror.com/deepmodelingcommunity/DPA-3.1-3M/resolve/main/DPA-3.1-3M.pt?download=true",
        ],
        "filename": "DPA-3.1-3M.pt",
        "sha256": "86dd3a804d78ca5d203ebf98747e8f16dff9713ba8950097ceb760b161e19907",
    },
}


def available_model_names() -> list[str]:
    """Return available model names from built-in registry."""
    return sorted(MODEL_REGISTRY)
