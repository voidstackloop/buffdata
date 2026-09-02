# buffdata AWS module

Provisions exactly the AWS-side pieces buffdata already knows how to talk to, scoped to
least privilege:

- an **S3 bucket** for datasets (`buffdata/models/formats.py`'s fsspec/s3fs backend
  reads/writes `s3://` URLs directly -- no separate sync step)
- a **Secrets Manager secret** for provider API keys (`buffdata/engine/secrets.py`'s
  `AWSSecretsManagerResolver`, selected via `BUFFDATA_SECRET_BACKEND=aws_secrets_manager`)
- an **IAM role**, trusted only by one Kubernetes ServiceAccount (IRSA) in one namespace,
  with permissions on nothing but that one bucket and that one secret

It assumes an EKS cluster (and its OIDC provider) already exists -- this module does not
create one. Wire the outputs into `deploy/helm/buffdata/values.yaml`:

| Terraform output              | Helm value                                  |
|--------------------------------|----------------------------------------------|
| `dataset_bucket_name`          | the `s3://<bucket>/...` URLs in `command`     |
| `provider_api_keys_secret_id`  | `awsSecretId` (with `secretBackend: aws_secrets_manager`) |
| `buffdata_role_arn`            | `serviceAccount.annotations["eks.amazonaws.com/role-arn"]` |

## Status

**Written against the real code paths, not verified.** No `terraform` binary was
available in the environment this was built in, so `terraform validate`/`plan` have not
been run -- and applying it would create real, billed AWS resources, which isn't
something to do without you reviewing the plan first regardless. Run:

```bash
terraform init
terraform validate
terraform plan -var="environment=staging" -var="eks_oidc_provider_arn=..." -var="eks_oidc_provider_url=..."
```

and read the plan before `apply`.
