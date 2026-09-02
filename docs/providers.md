# Providers, secrets, and storage

Three independent things to configure, and you only need the ones you're actually
using: which LLM `--provider`, where its credential comes from, and (separately)
whether your dataset files live locally or in cloud storage.

## LLM providers

| `--provider` | What it is | Model default |
|---|---|---|
| `gemini` | Google's public API | `gemini-3.7-flash` |
| `openai` | OpenAI's public API (Responses API) | `gpt-5.4-mini` |
| `anthropic` | Anthropic's public API (Messages API) | `claude-sonnet-4-6` |
| `azure_openai` | Your own Azure OpenAI resource -- request/response stay in your Azure tenant/region | none (pass your deployment name) |
| `bedrock_anthropic` | Anthropic via your own AWS Bedrock access | `us.anthropic.claude-sonnet-4-6-v1:0` |
| `openai_compatible` | Any server implementing the OpenAI chat-completions API: an internal gateway, LiteLLM, TGI, or anything not covered by a named local provider below | none (pass whatever the gateway exposes) |
| `ollama` / `lmstudio` / `vllm` / `llamacpp` | A local LLM server -- see below | none (pass the model name you pulled/loaded) |

Every provider is constructed by `create_llm_client()` in
[`buffdata/engine/client.py`](../buffdata/engine/client.py); see
[architecture.md](architecture.md#the-provider-neutral-client-contract) for how they
share one request/response contract.

### API keys

```env
GEMINI_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
BUFFDATA_PROVIDER=gemini
```

`bedrock_anthropic` uses standard AWS credential resolution (`AWS_ACCESS_KEY_ID` /
`AWS_PROFILE` / an instance role), not a BuffData-specific variable.

### Local LLM servers

Point BuffData at a model running on your own machine or another machine on your
team's network -- no data leaves your infrastructure, no per-token bill. Full detail,
worked examples, and the `--network-policy local` structural guarantee (calls can
never leave your network) are in the top-level
[README's "Local LLM servers" section](../README.md#local-llm-servers); in short:

```bash
buffdata score data.jsonl -o scored.jsonl --provider ollama --model llama3.1
buffdata score data.jsonl -o scored.jsonl --provider ollama \
  --base-url http://192.168.1.50:11434/v1 --model llama3.1   # a shared GPU box instead
```

| `--provider` | Default port | Override env var | API key env var (optional) |
|---|---|---|---|
| `ollama` | `11434` | `OLLAMA_BASE_URL` | `OLLAMA_API_KEY` |
| `lmstudio` | `1234` | `LMSTUDIO_BASE_URL` | `LMSTUDIO_API_KEY` |
| `vllm` | `8000` | `VLLM_BASE_URL` | `VLLM_API_KEY` |
| `llamacpp` | `8080` | `LLAMACPP_BASE_URL` | `LLAMACPP_API_KEY` |
| `openai_compatible` (generic) | none -- always required | `OPENAI_COMPATIBLE_BASE_URL` | `OPENAI_COMPATIBLE_API_KEY` |

Verified for real (not mocked) in
[`tests/test_local_llm.py`](../tests/test_local_llm.py): a minimal local HTTP server
speaking the actual OpenAI chat-completions wire format, with BuffData's real client
making genuine HTTP requests to it over loopback.

## Secret backends

API keys resolve through a swappable backend
([`buffdata/engine/secrets.py`](../buffdata/engine/secrets.py)) instead of
`os.getenv` directly, so a deployment can source every provider's credential from a
real secret manager without any client code changing:

```env
BUFFDATA_SECRET_BACKEND=env               # default -- plain environment variables
BUFFDATA_SECRET_BACKEND=vault             # HashiCorp Vault
BUFFDATA_SECRET_BACKEND=aws_secrets_manager
BUFFDATA_SECRET_BACKEND=gcp_secret_manager
BUFFDATA_SECRET_BACKEND=azure_key_vault
```

| Backend | Extra config | Secret shape expected |
|---|---|---|
| `vault` | `VAULT_ADDR`, `VAULT_TOKEN`, optional `BUFFDATA_VAULT_PATH` (default `buffdata`), `BUFFDATA_VAULT_MOUNT` (default `secret`) | One KV field per BuffData secret name |
| `aws_secrets_manager` | `BUFFDATA_AWS_SECRET_ID`, `AWS_REGION`/`AWS_DEFAULT_REGION`, standard AWS credentials | A single JSON secret: `{"GEMINI_API_KEY": "...", "ANTHROPIC_API_KEY": "...", ...}` |
| `gcp_secret_manager` | `GOOGLE_CLOUD_PROJECT`/`GCP_PROJECT` | One secret per BuffData secret name (literal name, e.g. `GEMINI_API_KEY`), latest version |
| `azure_key_vault` | `AZURE_KEY_VAULT_URL` | One secret per name, with underscores mapped to hyphens (Key Vault secret names can't contain underscores) -- `GEMINI_API_KEY` is looked up as `gemini-api-key` |

Requires `pip install -e ".[enterprise]"` for the non-`env` backends' SDKs
(`hvac`, `boto3`, `google-cloud-secret-manager`, `azure-keyvault-secrets`).

## Cloud dataset storage

Every focused single-stage command (`score`, `refine`, `evolve`, `dpo`, `dedup`,
`augment`, `scrub`, `classify`, `filter-ppl`) reads and writes `s3://`, `gs://`/`gcs://`,
and `az://`/`abfs://`/`abfss://`/`adl://` URLs directly via `fsspec`
([`buffdata/models/formats.py`](../buffdata/models/formats.py)):

```bash
buffdata score s3://my-bucket/data.jsonl -o s3://my-bucket/scored.jsonl --provider gemini
```

`optimize` and `pipeline` deliberately don't -- their checkpoint, fingerprint, and
`.rejected.jsonl`/`.report.json` sidecar artifacts assume a local filesystem. Download,
run one of those two, then upload; or use one of the cloud-URL-capable commands above
directly. Requires `pip install -e ".[enterprise]"` for `fsspec`/`s3fs`/`gcsfs`/`adlfs`.
