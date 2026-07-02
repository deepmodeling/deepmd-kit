# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Assembly property data: one user-facing facade, many sources.

An assembly sample is a set of weighted component frames sharing one label, plus
optional per-sample conditions (fparam).  The public API is intentionally small:

    from dpa_adapt import Assembly

    Assembly.from_polymer_csv("cp.csv", target="cloud_point").write(PATH)
    Assembly.mark_existing("oer/dpdata", target="overpotential")
    a = Assembly(target="x")
    a.sample(label=..., conditions={...}).add(coords, symbols, weight=...)
    a.write(PATH)

The DeepMD tensor names still use ``group_id`` internally because that is the
training primitive, but users describe assemblies and samples.
"""

_LAZY = {
    "Assembly": ("._core", "Assembly"),
    "ComponentSpec": ("._core", "ComponentSpec"),
    "PoolMask": ("._core", "PoolMask"),
    "SampleSpec": ("._core", "SampleSpec"),
    "SiteSelector": ("._core", "SiteSelector"),
    "SubstitutionSpec": ("._core", "SubstitutionSpec"),
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> object:
    if name in _LAZY:
        import importlib

        mod_name, attr_name = _LAZY[name]
        module = importlib.import_module(mod_name, __package__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
