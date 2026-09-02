# NOTE: written against the real integration points buffdata already supports --
# S3 storage (buffdata/models/formats.py's fsspec/s3fs backend) and the
# aws_secrets_manager SecretResolver (buffdata/engine/secrets.py, AWSSecretsManagerResolver,
# selected via BUFFDATA_SECRET_BACKEND=aws_secrets_manager + BUFFDATA_AWS_SECRET_ID) --
# but `terraform validate`/`plan` have not been run against it: no terraform binary was
# available in the environment this was written in, and applying it would create real
# AWS resources and billing. Run `terraform validate` and review a `terraform plan`
# yourself before applying.

locals {
  tags = merge(
    {
      Project     = "buffdata"
      Environment = var.environment
      ManagedBy   = "terraform"
    },
    var.tags,
  )
}

# --- Dataset storage -------------------------------------------------------------------

resource "aws_s3_bucket" "datasets" {
  bucket = "${var.name}-datasets-${var.environment}"
  tags   = local.tags
}

resource "aws_s3_bucket_versioning" "datasets" {
  bucket = aws_s3_bucket.datasets.id
  versioning_configuration {
    status = var.enable_bucket_versioning ? "Enabled" : "Disabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "datasets" {
  bucket = aws_s3_bucket.datasets.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "datasets" {
  bucket                  = aws_s3_bucket.datasets.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- Provider API keys -------------------------------------------------------------------

resource "aws_secretsmanager_secret" "provider_api_keys" {
  name = "${var.name}/provider-api-keys/${var.environment}"
  tags = local.tags
}

resource "aws_secretsmanager_secret_version" "provider_api_keys" {
  count         = length(var.provider_api_keys) > 0 ? 1 : 0
  secret_id     = aws_secretsmanager_secret.provider_api_keys.id
  secret_string = jsonencode(var.provider_api_keys)
}

# --- IAM: least-privilege role for the buffdata pod (IRSA) -----------------------------
# Trusts exactly one Kubernetes ServiceAccount (namespace + name), not the whole cluster.

data "aws_iam_policy_document" "assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [var.eks_oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "${var.eks_oidc_provider_url}:sub"
      values   = ["system:serviceaccount:${var.k8s_namespace}:${var.k8s_service_account_name}"]
    }

    condition {
      test     = "StringEquals"
      variable = "${var.eks_oidc_provider_url}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "buffdata_runner" {
  name               = "${var.name}-runner-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
  tags               = local.tags
}

data "aws_iam_policy_document" "buffdata_runner" {
  statement {
    sid    = "DatasetBucketReadWrite"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.datasets.arn,
      "${aws_s3_bucket.datasets.arn}/*",
    ]
  }

  statement {
    sid       = "ProviderApiKeysRead"
    effect    = "Allow"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.provider_api_keys.arn]
  }
}

resource "aws_iam_role_policy" "buffdata_runner" {
  name   = "${var.name}-runner-${var.environment}"
  role   = aws_iam_role.buffdata_runner.id
  policy = data.aws_iam_policy_document.buffdata_runner.json
}
