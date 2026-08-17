# Install with an AI agent

DeePMD-kit ships an official Agent Skill, `deepmd-install`, that walks an AI
agent through installing the package. The skill covers easy methods (conda,
pip, Docker, offline installers, `dp1s`) and source builds of the Python
package, the C++ interface, and LAMMPS with Kokkos (`pair_style deepmd/kk`,
used by DPA4/SeZM, and `pair_style dpa4spin/kk`, used by DPA4C).

The skill probes the machine, asks only for decisions required by the selected
installation path, and records concrete paths and versions in a validated
plan. It does not assume a host path, package mirror, CUDA toolkit, or backend
version.

The full skill catalog is in [Agent Skills](../agent-skills.md). For a
manual install, use [Easy install](easy-install.md) or
[Install from source](install-from-source.md).

## If you are a user

Send this page to your agent and ask it to install the official
`deepmd-install` skill, then install DeePMD-kit. You do not need to run the
commands below yourself.

## If you are an agent

Install `deepmd-install` first, then follow its probe, validated plan, and
gate workflow.

If this machine already has a DeePMD-kit checkout, run from the repository
root:

```bash
npx -y skills add ./skills --skill deepmd-install -y
```

If there is no checkout, install the same skill from GitHub:

```bash
npx -y skills add https://github.com/deepmodeling/deepmd-kit/tree/master/skills \
    --skill deepmd-install -y
```

If direct GitHub access fails, use `gh-proxy.com` for a public, read-only clone
and install from the local checkout. Do not send credentials or private
repository URLs through the proxy.

```bash
git clone --depth 1 \
    https://gh-proxy.com/https://github.com/deepmodeling/deepmd-kit.git \
    deepmd-kit-skill-source
npx -y skills add ./deepmd-kit-skill-source/skills \
    --skill deepmd-install -y
```

The examples require Node.js/npm so that `npx` is available. The Skills CLI
uses the detected agent. To target one product, add `--agent cursor`,
`--agent claude-code`, or another supported agent name. Refresh or restart the
session afterward so the skill is reloaded.

Then use the `deepmd-install` skill: probe the machine, ask only for missing
decisions, validate `install-plan.json`, and execute one gate at a time. A
CUDA LAMMPS build uses exactly one `Kokkos_ARCH_*` flag per binary and verifies
the pair styles required by the selected model family.
