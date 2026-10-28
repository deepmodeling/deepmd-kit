# Install with an AI agent

DeePMD-kit ships an official Agent Skill, `deepmd-install`, that walks an AI
agent through installing the package. The skill covers easy methods (conda,
pip, Docker, offline installers, `dp1s`) and source builds of the Python
package, the C++ interface, and LAMMPS.

The skill inspects the target machine, asks only for decisions required by the
selected installation path, reads the official documentation matching the
requested version, and verifies the requested interface. Version-specific
package and build commands remain in the installation documentation instead
of being duplicated in the skill.

The full skill catalog is in [Agent Skills](../agent-skills.md). For a
manual install, use [Easy install](easy-install.md) or
[Install from source](install-from-source.md).

## If you are a user

Send this page to your agent and ask it to install the official
`deepmd-install` skill, then install DeePMD-kit. You do not need to run the
commands below yourself.

## If you are an agent

Install `deepmd-install` first, then load it and install DeePMD-kit with the
required backend, accelerator, and optional native interfaces.

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

If direct GitHub access fails, clone the official Gitee mirror and install from
the local checkout:

```bash
git clone --depth 1 \
    https://gitee.com/deepmodeling/deepmd-kit.git \
    deepmd-kit-skill-source
npx -y skills add ./deepmd-kit-skill-source/skills \
    --skill deepmd-install -y
```

The examples require Node.js/npm so that `npx` is available. The Skills CLI
uses the detected agent. To target one product, add `--agent cursor`,
`--agent claude-code`, or another supported agent name. Refresh or restart the
session afterward so the skill is reloaded.

Then use the `deepmd-install` skill. The agent selects one installation method,
loads the version-matched official documentation, executes its commands in the
chosen environment, and verifies the requested Python, C/C++, or LAMMPS
interface.
