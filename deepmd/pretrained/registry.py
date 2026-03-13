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
            "https://modelscope.cn/models/DeepModelingCommunity/DPA-3.2-5M/resolve/master/DPA-3.2-5M.pt",
        ],
        "filename": "DPA-3.2-5M.pt",
        "sha256": "876354744aeaae17b2639a6a690514470273784f2b4836280850f50cbb799165",
    },
    "DPA-3.1-3M": {
        "urls": [
            "https://huggingface.co/deepmodelingcommunity/DPA-3.1-3M/resolve/main/DPA-3.1-3M.pt?download=true",
            "https://hf-mirror.com/deepmodelingcommunity/DPA-3.1-3M/resolve/main/DPA-3.1-3M.pt?download=true",
            "https://modelscope.cn/models/DeepModelingCommunity/DPA-3.1-3M/resolve/master/DPA-3.1-3M.pt",
        ],
        "filename": "DPA-3.1-3M.pt",
        "sha256": "86dd3a804d78ca5d203ebf98747e8f16dff9713ba8950097ceb760b161e19907",
    },
    "DPA-2.4-7M": {
        "urls": [
            "https://huggingface.co/deepmodelingcommunity/DPA-2.4-7M/resolve/main/dpa-2.4-7M.pt?download=true",
            "https://hf-mirror.com/deepmodelingcommunity/DPA-2.4-7M/resolve/main/dpa-2.4-7M.pt?download=true",
            "https://modelscope.cn/models/DeepModelingCommunity/DPA-2.4-7M/resolve/master/dpa-2.4-7M.pt",
        ],
        "filename": "dpa-2.4-7M.pt",
        "sha256": "7a5ca2b01579d9617502b4203af839107fdcf1ec7e3ae1d66a5b14811bc5b741",
    },
}


def available_model_names() -> list[str]:
    """Return available model names from built-in registry."""
    return sorted(MODEL_REGISTRY)
