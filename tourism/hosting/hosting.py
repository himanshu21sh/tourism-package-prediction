from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id="himanshu21sh/tourism-package-prediction",
repo_type="space"

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")


api.upload_folder(
    folder_path="tourism/deployment",
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="" 
    )
