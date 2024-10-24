# SPDX-License-Identifier: LGPL-3.0-or-later
import paddle
import torch

psd = torch.load(
    "/workspace/hesensen/deepmd_backend/deepmd_paddle_new/source/tests/pd/model/models/dpa1.pth",
    "cpu",
)

tsd = {}
for k, v in psd.items():
    # if ".matrix" in k:
    #     v = v.T
    psd[k] = paddle.to_tensor(v.detach().cpu().numpy())

paddle.save(
    psd,
    "/workspace/hesensen/deepmd_backend/deepmd_paddle_new/source/tests/pd/model/models/dpa1.pdparams",
)
