variable "name" {
  description = "Prefix for every resource this module creates (bucket, secret, role)."
  type        = string
  default     = "buffdata"
}

variable "environment" {
  description = "Tag applied to every resource, e.g. \"prod\", \"staging\"."
  type        = string
}

variable "enable_bucket_versioning" {
  description = "Keep prior dataset versions in S3. Costs storage; recommended in prod so a bad optimize/pipeline run can be rolled back by re-pointing at the previous object version."
  type        = bool
  default     = true
}

variable "provider_api_keys" {
  description = <<-EOT
    Provider API keys to seed into the Secrets Manager secret, keyed exactly as
    buffdata/engine/client.py resolves them (GEMINI_API_KEY, OPENAI_API_KEY,
    ANTHROPIC_API_KEY, ...). Leave empty and populate the secret out-of-band
    (e.g. via a separate rotation process) if you'd rather Terraform state never
    hold key material -- this module creates the secret either way.
  EOT
  type        = map(string)
  default     = {}
  sensitive   = true
}

variable "eks_oidc_provider_arn" {
  description = <<-EOT
    ARN of the existing EKS cluster's IAM OIDC provider (aws eks describe-cluster,
    or the `oidc_provider_arn` output of most EKS Terraform modules). This module
    does not create an EKS cluster or its OIDC provider -- it only trusts one that
    already exists, scoped to one Kubernetes ServiceAccount.
  EOT
  type        = string
}

variable "eks_oidc_provider_url" {
  description = "Issuer URL of the same OIDC provider, without the https:// prefix (e.g. oidc.eks.us-east-1.amazonaws.com/id/XXXXXXXX)."
  type        = string
}

variable "k8s_namespace" {
  description = "Kubernetes namespace the buffdata Helm release runs in."
  type        = string
  default     = "default"
}

variable "k8s_service_account_name" {
  description = "Name of the ServiceAccount the buffdata Helm release uses (serviceAccount.name in deploy/helm/buffdata/values.yaml, or the chart's default \"<release>-buffdata\" if left unset there)."
  type        = string
  default     = "buffdata"
}

variable "tags" {
  description = "Extra tags merged onto every resource."
  type        = map(string)
  default     = {}
}
