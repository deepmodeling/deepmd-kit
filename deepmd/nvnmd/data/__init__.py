# SPDX-License-Identifier: LGPL-3.0-or-later
"""nvnmd.data
==========.

Provides
    1. hardware configuration
    2. default input script
    3. title and citation

Data
----

jdata_sys
    action configuration
jdata_config
    hardware configuration

    dscp
        descriptor configuration
    fitn
        fitting network configuration
    size
        ram capacity
    ctrl
        control flag, such as Time Division Multiplexing (TDM)
    nbit
        number of bits of fixed-point number
jdata_config_16 (disable)
    difference with configure fitting size as 16
jdata_config_32 (disable)
    difference with configure fitting size as 32
jdata_config_64 (disable)
    difference with configure fitting size as 64
jdata_config_128 (default)
    difference with configure fitting size as 128
jdata_configs
    all configure of jdata_config{nfit_node}
jdata_deepmd_input
    default input script for nvnmd training
NVNMD_WELCOME
    nvnmd title when logging
NVNMD_CITATION
    citation of nvnmd
"""
