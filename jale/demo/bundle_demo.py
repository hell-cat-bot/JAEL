#!/usr/bin/env python3
"""Inline demo/demo_data.json into demo/index.html -> demo/jale_demo.html.

The result is a single self-contained file: no server, no fetch, no build step
at view time. That is the file to hand a judge or publish.

    python -m jale.demo.build_demo_data --profile SMOKE
    python -m jale.demo.bundle_demo
"""
from __future__ import annotations

import json
from pathlib import Path

SRC = Path("demo/index.html")
DATA = Path("demo/demo_data.json")
OUT = Path("demo/jale_demo.html")


def main() -> None:
    html = SRC.read_text(encoding="utf-8")
    if "__DATA__" not in html:
        raise SystemExit("demo/index.html has no __DATA__ placeholder")
    data = json.loads(DATA.read_text(encoding="utf-8"))
    # compact, and neutralise any '<' so the JSON cannot terminate the <script>
    blob = json.dumps(data, separators=(",", ":"), default=str).replace("<", "\\u003c")
    OUT.write_text(html.replace("__DATA__", blob), encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
