{{- define "buffdata.name" -}}
buffdata
{{- end -}}

{{- define "buffdata.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "buffdata.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "buffdata.labels" -}}
app.kubernetes.io/name: {{ include "buffdata.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "buffdata.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "buffdata.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Shared env for the buffdata container: provider API keys (only set when
secretBackend: env), plus the secret-backend selector and its backend-specific
connection settings (buffdata/engine/secrets.py resolves these at runtime).
*/}}
{{- define "buffdata.env" -}}
- name: BUFFDATA_SECRET_BACKEND
  value: {{ .Values.secretBackend | quote }}
{{- if eq .Values.secretBackend "env" }}
{{- if .Values.apiKeys.geminiApiKey }}
- name: GEMINI_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "buffdata.fullname" . }}-api-keys
      key: gemini-api-key
{{- end }}
{{- if .Values.apiKeys.openaiApiKey }}
- name: OPENAI_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "buffdata.fullname" . }}-api-keys
      key: openai-api-key
{{- end }}
{{- if .Values.apiKeys.anthropicApiKey }}
- name: ANTHROPIC_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "buffdata.fullname" . }}-api-keys
      key: anthropic-api-key
{{- end }}
{{- end }}
{{- if eq .Values.secretBackend "aws_secrets_manager" }}
- name: BUFFDATA_AWS_SECRET_ID
  value: {{ .Values.awsSecretId | quote }}
- name: AWS_REGION
  value: {{ .Values.awsRegion | quote }}
{{- end }}
{{- end -}}

{{/*
Pod template shared by job.yaml and cronjob.yaml -- kept in one place so the two
trigger modes (on-demand vs scheduled) can never drift apart on the part that actually
runs buffdata.
*/}}
{{- define "buffdata.podTemplate" -}}
metadata:
  labels:
    {{- include "buffdata.labels" . | nindent 4 }}
spec:
  restartPolicy: Never
  serviceAccountName: {{ include "buffdata.serviceAccountName" . }}
  {{- with .Values.nodeSelector }}
  nodeSelector:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  {{- with .Values.affinity }}
  affinity:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  {{- with .Values.tolerations }}
  tolerations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  containers:
    - name: buffdata
      image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
      imagePullPolicy: {{ .Values.image.pullPolicy }}
      args:
        {{- toYaml .Values.command | nindent 8 }}
      env:
        {{- include "buffdata.env" . | nindent 8 }}
      resources:
        {{- toYaml .Values.resources | nindent 8 }}
      {{- if .Values.pipelineConfig.enabled }}
      volumeMounts:
        - name: pipeline-config
          mountPath: /config
          readOnly: true
      {{- end }}
  {{- if .Values.pipelineConfig.enabled }}
  volumes:
    - name: pipeline-config
      configMap:
        name: {{ include "buffdata.fullname" . }}-config
  {{- end }}
{{- end -}}
