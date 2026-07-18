# SPDX-License-Identifier: LGPL-3.0-or-later
"""Lock DPATrainer._build_fitting_net's dim_case_embd behavior.

History (the "repeatedly reverted" patch): 2026-05-18 a dim_case_embd=31
injection was added for FT/LP, because `--finetune --model-branch <branch>`
tried to copy the branch's [159, 240] property head and failed without it.
On 2026-05-20 the FT/LP command was realigned to the paper repo, which uses
`--finetune` WITHOUT --model-branch: single-task fine-tune copies only the
backbone and random-inits the property head at [128, 240]. With no branch
head to size-match, dim_case_embd must NOT be injected (the paper qm9_gap
input.json omits it).

So: FT/LP fitting_net has no dim_case_embd unless the user sets it
explicitly via fitting_net_params. These tests build config only.
"""

from __future__ import (
    annotations,
)

from dpa_adapt.trainer import (
    DPATrainer,
)

TYPE_MAP = ["H", "C", "N", "O"]
DUMMY_SYS = ["/data/sys"]


def _trainer(pretrained, **overrides):
    kwargs = {
        "pretrained": pretrained,
        "train_systems": DUMMY_SYS,
        "valid_systems": DUMMY_SYS,
        "type_map": TYPE_MAP,
    }
    kwargs.update(overrides)
    return DPATrainer(**kwargs)


def test_pretrained_mode_no_dim_case_embd(tmp_path):
    """FT/LP (pretrained != None) must NOT inject dim_case_embd: the paper
    single-task fine-tune random-inits the property head, so there is no
    [159, 240] checkpoint head to match.
    """
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"")
    t = _trainer(str(ckpt))
    fn = t._build_fitting_net()
    assert fn.get("dim_case_embd") is None


def test_user_fitting_net_params_can_set_dim_case_embd(tmp_path):
    """An explicit user-supplied dim_case_embd is still honored verbatim."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"")
    t = _trainer(str(ckpt), fitting_net_params={"dim_case_embd": 99})
    fn = t._build_fitting_net()
    assert fn["dim_case_embd"] == 99
