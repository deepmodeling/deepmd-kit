# SPDX-License-Identifier: LGPL-3.0-or-later
"""Registry of built-in pretrained model sources."""

from typing import (
    Any,
)

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "DPA-3.3-1M": {
        "urls": [
            "https://huggingface.co/deepmodelingcommunity/DPA-3.3-1M/resolve/main/DPA-3.3-1M.pt?download=true",
            "https://hf-mirror.com/deepmodelingcommunity/DPA-3.3-1M/resolve/main/DPA-3.3-1M.pt?download=true",
            "https://modelscope.cn/models/DeepModelingCommunity/DPA-3.3-1M/resolve/master/DPA-3.3-1M.pt",
        ],
        "filename": "DPA-3.3-1M.pt",
        "sha256": "36fe440c111108d60cda54aa7d3fccac743794de25abef4d49564b9fb896a55b",
    },
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
            "https://huggingface.co/deepmodelingcommunity/DPA-2.4-7M/resolve/main/DPA-2.4-7M-patched-mt.pt?download=true",
            "https://hf-mirror.com/deepmodelingcommunity/DPA-2.4-7M/resolve/main/DPA-2.4-7M-patched-mt.pt?download=true",
            "https://modelscope.cn/models/DeepModelingCommunity/DPA-2.4-7M/resolve/master/DPA-2.4-7M-patched-mt.pt",
        ],
        "filename": "dpa-2.4-7M.pt",
        "sha256": "904eb5560af9ff644347dedd3ebf1e9c97929d02ee37ce3cbe895de3df711198",
    },
    "DPA3-Omol-Large": {
        "urls": [
            "https://huggingface.co/deepmodelingcommunity/DPA3-Omol-Large/resolve/main/DPA3-Omol-Large.pt?download=true",
            "https://hf-mirror.com/deepmodelingcommunity/DPA3-Omol-Large/resolve/main/DPA3-Omol-Large.pt?download=true",
            "https://modelscope.cn/models/DeepModelingCommunity/DPA3-Omol-Large/resolve/master/DPA3-Omol-Large.pt",
        ],
        "filename": "DPA3-Omol-Large.pt",
        "sha256": "dc4d252b31450b41eb3546cc48f640ad0831c0b5d069ce27d996e0ff58fc037a",
    },
    "DPA4-Nano-OMat24-v20260805": {
        "urls": [
            "https://huggingface.co/deepmodelingcommunity/DPA4-OMat24/resolve/main/DPA4-Nano-OMat24-v20260805.pt?download=true",
            "https://hf-mirror.com/deepmodelingcommunity/DPA4-OMat24/resolve/main/DPA4-Nano-OMat24-v20260805.pt?download=true",
            "https://modelscope.cn/models/DeepModelingCommunity/DPA4-OMat24/resolve/master/DPA4-Nano-OMat24-v20260805.pt",
        ],
        "filename": "DPA4-Nano-OMat24-v20260805.pt",
        "sha256": "ded047546c2b44a50e2c850680d77279e2bf3e107c10f1c2fe39a5c344e693f6",
    },
    "DPA4-Mini-OMat24-v20260805": {
        "urls": [
            "https://huggingface.co/deepmodelingcommunity/DPA4-OMat24/resolve/main/DPA4-Mini-OMat24-v20260805.pt?download=true",
            "https://hf-mirror.com/deepmodelingcommunity/DPA4-OMat24/resolve/main/DPA4-Mini-OMat24-v20260805.pt?download=true",
            "https://modelscope.cn/models/DeepModelingCommunity/DPA4-OMat24/resolve/master/DPA4-Mini-OMat24-v20260805.pt",
        ],
        "filename": "DPA4-Mini-OMat24-v20260805.pt",
        "sha256": "13edeea63448c0f8a0b38e08fd8ba196fb1e2fad5f12486b1cd23320829da268",
    },
    "DPA4-Neo-OMat24-v20260805": {
        "urls": [
            "https://huggingface.co/deepmodelingcommunity/DPA4-OMat24/resolve/main/DPA4-Neo-OMat24-v20260805.pt?download=true",
            "https://hf-mirror.com/deepmodelingcommunity/DPA4-OMat24/resolve/main/DPA4-Neo-OMat24-v20260805.pt?download=true",
            "https://modelscope.cn/models/DeepModelingCommunity/DPA4-OMat24/resolve/master/DPA4-Neo-OMat24-v20260805.pt",
        ],
        "filename": "DPA4-Neo-OMat24-v20260805.pt",
        "sha256": "fd7f34ae28f921201e4a0328ddf56179892752084f8363e7796038201471c989",
    },
    "DPA4-Air-OMat24-v20260805": {
        "urls": [
            "https://huggingface.co/deepmodelingcommunity/DPA4-OMat24/resolve/main/DPA4-Air-OMat24-v20260805.pt?download=true",
            "https://hf-mirror.com/deepmodelingcommunity/DPA4-OMat24/resolve/main/DPA4-Air-OMat24-v20260805.pt?download=true",
            "https://modelscope.cn/models/DeepModelingCommunity/DPA4-OMat24/resolve/master/DPA4-Air-OMat24-v20260805.pt",
        ],
        "filename": "DPA4-Air-OMat24-v20260805.pt",
        "sha256": "7ac5fb696fc057229ceebc33f31f7a96b2f3616ac5b63a512acd1848f2d41a09",
    },
    "DPA4-Plus-OMat24-v20260805": {
        "urls": [
            "https://huggingface.co/deepmodelingcommunity/DPA4-OMat24/resolve/main/DPA4-Plus-OMat24-v20260805.pt?download=true",
            "https://hf-mirror.com/deepmodelingcommunity/DPA4-OMat24/resolve/main/DPA4-Plus-OMat24-v20260805.pt?download=true",
            "https://modelscope.cn/models/DeepModelingCommunity/DPA4-OMat24/resolve/master/DPA4-Plus-OMat24-v20260805.pt",
        ],
        "filename": "DPA4-Plus-OMat24-v20260805.pt",
        "sha256": "6820a320dc241a4002e6c257ad21983950df178b1f80271a6e6af199adb18567",
    },
}


def available_model_names() -> list[str]:
    """Return available model names from built-in registry."""
    return sorted(MODEL_REGISTRY)
