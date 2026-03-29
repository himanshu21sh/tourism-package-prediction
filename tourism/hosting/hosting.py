from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism/deployment",
    repo_id="himanshu21sh/tourism-package-prediction",
    repo_type="space",
    path_in_repo="" 
    )
