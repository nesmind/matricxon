"""Manual smoke test (M7): a `/api/pull` against a real Hugging Face repo: `HFRepoResolver` suffix
matching, a streaming download, sha256 verification and a sidecar from a header-only GGUF parse. Not
pass/fail; needs network (~19MB by default).

    .venv/bin/python -m tools.manual_pull_check
"""

import argparse
import json
import tempfile
from pathlib import Path

from app.models.catalog import ModelCatalog
from app.pull.job import PullJob


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="hf.co/leliuga/all-MiniLM-L6-v2-GGUF:Q2_K")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir:
        models_dir = Path(tmp_dir)
        job = PullJob(models_dir)

        for line in job.run(args.tag):
            print(json.dumps(line))
            if line.get("error"):
                return

        installed = ModelCatalog(models_dir).get(args.tag)
        print(f"\ninstalled: {installed}")


if __name__ == "__main__":
    main()
