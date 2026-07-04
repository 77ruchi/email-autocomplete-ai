from huggingface_hub import HfApi

api = HfApi()

api.create_repo("ruchita77/fine-tuned-gpt2-email", repo_type="model", exist_ok=True)

api.upload_folder(
    folder_path="models/fine_tuned_gpt2",
    repo_id="ruchita77/fine-tuned-gpt2-email",
    repo_type="model"
)

print("Upload complete!")