"""Verify publication bytes only; no experiment imports, tensors, or remote calls."""
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parent
m = json.loads((root / "manifest.json").read_text())
actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p != root / "manifest.json"}
assert actual == set(m["files"]), "Publication file inventory differs"
for name, record in m["files"].items():
    b = (root / name).read_bytes()
    assert len(b) == record["bytes"] and hashlib.sha256(b).hexdigest() == record["sha256"], name
    if name.endswith(".json"):
        json.loads(b)
print("Verified", len(actual), "publication files; no numerical audit executed.")
