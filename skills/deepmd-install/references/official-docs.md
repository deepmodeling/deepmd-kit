# Official documentation routing

Fetch documentation before choosing version-sensitive packages, image tags,
build variables, or LAMMPS options. This reference assumes there is no local
DeePMD-kit checkout.

## Fetch procedure

1. Use the agent's browser, URL reader, or web-fetch tool to open a direct URL
   from this file. Read the page body and relevant tabs; a search snippet is
   not documentation.

1. Determine the target release tag or Git ref before selecting commands. For
   a stable release, obtain the tag from the official Releases page.

1. Open the versioned documentation. If it is missing or incomplete, retrieve
   the corresponding raw Markdown from the exact tag or commit.

1. If the agent has no web-fetch tool, use `curl` and keep the response in the
   tool output:

   ```bash
   curl -fsSL --retry 2 \
       "https://raw.githubusercontent.com/deepmodeling/deepmd-kit/<REF>/doc/install/<FILE>.md"
   ```

1. Stop on a nonzero curl exit, empty response, or HTML error page. Do not
   generate commands from partial or unverified content.

## Primary pages

- Easy install: <https://docs.deepmodeling.com/projects/deepmd/en/latest/getting-started/install.html>
- Source Python and C/C++: <https://docs.deepmodeling.com/projects/deepmd/en/latest/install/install-from-source.html>
- Pre-compiled C library: <https://docs.deepmodeling.com/projects/deepmd/en/latest/install/install-from-c-library.html>
- LAMMPS: <https://docs.deepmodeling.com/projects/deepmd/en/latest/install/install-lammps.html>
- Development packages: <https://docs.deepmodeling.com/projects/deepmd/en/latest/install/easy-install-dev.html>
- Releases: <https://github.com/deepmodeling/deepmd-kit/releases>
- Container images: <https://github.com/deepmodeling/deepmd-kit/pkgs/container/deepmd-kit>
- dp1s options: <https://github.com/deepmodeling-activity/dp1s>

Use official backend installers when the DeePMD-kit page links to them:

- PyTorch: <https://pytorch.org/get-started/locally/>
- TensorFlow: <https://www.tensorflow.org/install>
- JAX: <https://docs.jax.dev/en/latest/installation.html>
- Paddle: <https://www.paddlepaddle.org.cn/install/quick>

## Match the target version

The `latest` documentation follows the development branch and may not describe
an older release. For a release, first try the versioned documentation root:

```text
https://docs.deepmodeling.com/projects/deepmd/en/v<VERSION>/
```

If that build is unavailable or incomplete, fetch the relevant Markdown file
from the exact tag or commit in the official repository. Typical paths are:

```text
doc/install/easy-install.md
doc/install/easy-install-dev.md
doc/install/install-from-source.md
doc/install/install-from-c-library.md
doc/install/install-lammps.md
pyproject.toml
```

For example, replace `<REF>` with a validated tag or commit in either form:

```text
https://github.com/deepmodeling/deepmd-kit/blob/<REF>/doc/install/install-from-source.md
https://raw.githubusercontent.com/deepmodeling/deepmd-kit/<REF>/doc/install/install-from-source.md
```

Resolve a branch or tag to a commit SHA before building. Treat a Git ref as
data: reject values beginning with `-`, quote it, and use Git's `--` option
boundary where supported.

## When a page lacks the required detail

Inspect authoritative files at the same ref instead of borrowing commands from
another version:

1. `pyproject.toml` for Python requirements and extras;
1. `source/CMakeLists.txt` and included CMake modules for current build options;
1. `.github/workflows/` and `source/install/docker/Dockerfile` for maintained
   package and image build examples;
1. the selected LAMMPS tree for available Kokkos architecture names.

State clearly when a conclusion is inferred from maintained build
configuration rather than stated in the user documentation. Do not invent a
version, wheel index, image tag, CMake option, or Kokkos architecture.

## Network fallback

If a GitHub repository or raw-file URL is unreachable because of network
restrictions, retry the same public URL through `gh-proxy.com`, for example:

```text
https://gh-proxy.com/https://github.com/deepmodeling/deepmd-kit.git
https://gh-proxy.com/https://raw.githubusercontent.com/deepmodeling/deepmd-kit/<REF>/doc/install/install-from-source.md
```

Use the proxy only for public GitHub content. Never send credentials through
it. After cloning through a proxy, verify the configured upstream URL and the
resolved commit SHA before building.
