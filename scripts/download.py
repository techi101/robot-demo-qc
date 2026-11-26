"""Download only the parquet + metadata files of each dataset in datasets.json, at the pinned revision."""
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["data/chunk-000/file-000.parquet", "meta/episodes/chunk-000/file-000.parquet",
         "meta/info.json", "meta/stats.json"]

for src in json.loads((ROOT / "datasets.json").read_text()):
    api = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{src['repo']}/revision/{src['revision']}"))
    lic = [t for t in api.get("tags", []) if t.startswith("license:")]
    assert not api.get("gated") and f"license:{src['license']}" in lic, (src["repo"], api.get("gated"), lic)
    name = src["repo"].split("/")[-1]
    for f in FILES:
        out = ROOT / "raw" / name / f
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            urllib.request.urlretrieve(f"https://huggingface.co/datasets/{src['repo']}/resolve/{src['revision']}/{f}", out)
    print(f"{src['repo']}@{src['revision'][:7]}  {lic[0]}  not gated  ok")
