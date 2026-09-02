output "dataset_bucket_name" {
  description = "Pass as s3://<this>/... in buffdata's --input/--output or the Helm chart's `command`."
  value       = aws_s3_bucket.datasets.id
}

output "provider_api_keys_secret_id" {
  description = "Set as BUFFDATA_AWS_SECRET_ID (values.awsSecretId in the Helm chart)."
  value       = aws_secretsmanager_secret.provider_api_keys.id
}

output "buffdata_role_arn" {
  description = "Set as the eks.amazonaws.com/role-arn annotation on the Helm chart's ServiceAccount (values.serviceAccount.annotations)."
  value       = aws_iam_role.buffdata_runner.arn
}
