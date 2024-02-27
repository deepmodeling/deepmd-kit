# SPDX-License-Identifier: LGPL-3.0-or-later
import logging

log = logging.getLogger(__name__)


def update_one_sel(jdata, descriptor, mixed_type: bool = False):
    rcut = descriptor["rcut"]
    tmp_sel = get_sel(
        jdata,
        rcut,
        mixed_type=mixed_type,
    )
    sel = descriptor["sel"]
    if isinstance(sel, int):
        # convert to list and finnally convert back to int
        sel = [sel]
    if parse_auto_sel(descriptor["sel"]):
        ratio = parse_auto_sel_ratio(descriptor["sel"])
        descriptor["sel"] = sel = [int(wrap_up_4(ii * ratio)) for ii in tmp_sel]
    else:
        # sel is set by user
        for ii, (tt, dd) in enumerate(zip(tmp_sel, sel)):
            if dd and tt > dd:
                # we may skip warning for sel=0, where the user is likely
                # to exclude such type in the descriptor
                log.warning(
                    "sel of type %d is not enough! The expected value is "
                    "not less than %d, but you set it to %d. The accuracy"
                    " of your model may get worse." % (ii, tt, dd)
                )
    if mixed_type:
        descriptor["sel"] = sel = sum(sel)
    return descriptor


def parse_auto_sel(sel):
    if not isinstance(sel, str):
        return False
    words = sel.split(":")
    if words[0] == "auto":
        return True
    else:
        return False


def parse_auto_sel_ratio(sel):
    if not parse_auto_sel(sel):
        raise RuntimeError(f"invalid auto sel format {sel}")
    else:
        words = sel.split(":")
        if len(words) == 1:
            ratio = 1.1
        elif len(words) == 2:
            ratio = float(words[1])
        else:
            raise RuntimeError(f"invalid auto sel format {sel}")
        return ratio


def wrap_up_4(xx):
    return 4 * ((int(xx) + 3) // 4)


def get_sel(jdata, rcut, mixed_type: bool = False):
    _, max_nbor_size = get_nbor_stat(jdata, rcut, mixed_type=mixed_type)
    return max_nbor_size
