"""One-shot: create the HF repo and upload the model card + code snapshot.
Token is read from HF_TOKEN env var (never written to disk)."""
import os
from huggingface_hub import HfApi

REPO = "piyushptiwari/bytical-talk"
api = HfApi(token=os.environ["HF_TOKEN"])

api.create_repo(REPO, repo_type="model", exist_ok=True, private=False)

# Model card becomes the repo README.
api.upload_file(path_or_fileobj="MODEL_CARD.md", path_in_repo="README.md",
                repo_id=REPO, repo_type="model")

# Upload the brain/code package + docs so the repo is self-describing.
for folder, pat in [("bytical_talk", "**/*.py")]:
    api.upload_folder(folder_path=folder, path_in_repo=folder, repo_id=REPO,
                      repo_type="model", allow_patterns=[pat])
for f in ["README.md", "ROADMAP.md", "PLAN_C.md", "pyproject.toml",
          ".env.example", "LICENSE", "configs/default.yaml",
          "scripts/fetch_upstream.sh"]:
    api.upload_file(path_or_fileobj=f, path_in_repo=f, repo_id=REPO, repo_type="model")

print("HF_PUSH_OK https://huggingface.co/" + REPO)
