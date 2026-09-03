import polars as pl
from datasets import Dataset

def push_to_hub(jsonl_path: str, repo_id: str, token: str = None, private: bool = True):
    from buffdata.security.policy import check_network_url, check_input, remember_secret
    from buffdata.engine.secrets import get_default_secret_resolver
    check_input(jsonl_path)
    check_network_url("https://huggingface.co")
    token = remember_secret(token or get_default_secret_resolver().get("HF_TOKEN"))
    print(f"Loading {jsonl_path} via Polars...")
    df = pl.read_ndjson(jsonl_path)
    print("Converting to HuggingFace Dataset format...")
    hf_dataset = Dataset.from_pandas(df.to_pandas())
    print(f"Pushing to HF Hub: {repo_id}...")
    hf_dataset.push_to_hub(repo_id, token=token, private=private)
    print("Push complete!")
