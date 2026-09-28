#!/usr/bin/env python3
"""CI assertion: `decide --router` must produce a usable route field.
Reads one decide JSON on stdin; exits non-zero unless the route carries an
anchor, a confidence, and a mapped model (BUG #3 regression gate)."""
import json
import sys

d = json.load(sys.stdin)
r = d.get("route", {})
ok = (r.get("anchor") in {"rt%02d" % i for i in range(16)}
      and isinstance(r.get("confidence"), (int, float))
      and bool(r.get("model")))
if not ok:
    print(f"route assertion FAILED: {json.dumps(r)}", file=sys.stderr)
    sys.exit(2)
print(f"route OK: {r['anchor']} conf={r['confidence']} -> {r['model']}")
