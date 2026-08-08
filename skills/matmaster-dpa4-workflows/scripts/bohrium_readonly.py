#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Read-only Bohrium OpenAPI probes using only Python's standard library."""

# ruff: noqa: T201 -- stdout is this command-line tool's result interface.

from __future__ import (
    annotations,
)

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request

BASE = os.environ.get("BOHR_OPENAPI_BASE", "https://open.bohrium.com/openapi").rstrip(
    "/"
)


def get(path: str, params: dict | None = None) -> dict:
    access_key = os.environ.get("BOHR_ACCESS_KEY", "")
    if not access_key:
        raise SystemExit("BOHR_ACCESS_KEY is not set")
    query = urllib.parse.urlencode(
        {key: value for key, value in (params or {}).items() if value is not None}
    )
    url = f"{BASE}{path}" + (f"?{query}" if query else "")
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {access_key}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {body[:1000]}") from exc
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SystemExit(f"request failed: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("identity")
    sub.add_parser("projects")
    p = sub.add_parser("jobs")
    p.add_argument("--project-id", type=int)
    p.add_argument("--status", type=int, choices=(-1, 0, 1, 2, 3))
    p.add_argument("--page", type=int, default=1)
    p.add_argument("--page-size", type=int, default=20)
    sub.add_parser("node-resources")
    p = sub.add_parser("nodes")
    p.add_argument("--page", type=int, default=1)
    p.add_argument("--page-size", type=int, default=20)
    p = sub.add_parser("image-search")
    p.add_argument("keyword")
    p.add_argument("--limit", type=int, default=10)
    p = sub.add_parser("file-stat")
    p.add_argument("path")
    p.add_argument("--project-id", type=int, required=True)
    p.add_argument("--user-id", type=int, required=True)
    args = parser.parse_args()

    if args.action == "identity":
        result = get("/v1/ak/get")
    elif args.action == "projects":
        result = get("/v2/project/lite_list")
    elif args.action == "jobs":
        result = get(
            "/v1/job/list",
            {
                "projectId": args.project_id,
                "status": args.status,
                "page": args.page,
                "pageSize": args.page_size,
            },
        )
    elif args.action == "node-resources":
        result = get("/v2/node/resources")
    elif args.action == "nodes":
        result = get("/v2/node/list", {"page": args.page, "pageSize": args.page_size})
    elif args.action == "image-search":
        result = get(
            "/v2/image/public/version/search",
            {"keyword": args.keyword, "page": 1, "pageSize": args.limit},
        )
    else:
        encoded = urllib.parse.quote(args.path.lstrip("/"), safe="/")
        result = get(
            f"/v1/file/stat/{encoded}",
            {"projectId": args.project_id, "userId": args.user_id},
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
