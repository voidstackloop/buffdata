import json
from html import escape
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from buffdata.models.schemas import DatasetItem

class ReportGenerator:
    """Generates rich HTML and Markdown dataset optimization audit reports."""

    @staticmethod
    def compute_stats(items: List[DatasetItem]) -> Dict:
        scores = [it.quality_score.overall_score for it in items if it.quality_score]
        clarities = [it.quality_score.clarity for it in items if it.quality_score]
        accuracies = [it.quality_score.factual_accuracy for it in items if it.quality_score]
        depths = [it.quality_score.reasoning_depth for it in items if it.quality_score]
        instructions = [it.quality_score.instruction_following for it in items if it.quality_score]

        all_issues = []
        for it in items:
            if it.quality_score and it.quality_score.issues:
                all_issues.extend(it.quality_score.issues)

        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

        return {
            "total_items": len(items),
            "scored_items": len(scores),
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "median_score": float(np.median(scores)) if scores else 0.0,
            "min_score": float(np.min(scores)) if scores else 0.0,
            "max_score": float(np.max(scores)) if scores else 0.0,
            "std_score": float(np.std(scores)) if scores else 0.0,
            "mean_clarity": float(np.mean(clarities)) if clarities else 0.0,
            "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
            "mean_depth": float(np.mean(depths)) if depths else 0.0,
            "mean_instruction": float(np.mean(instructions)) if instructions else 0.0,
            "top_issues": sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:8],
        }

    @classmethod
    def generate_markdown_summary(cls, items: List[DatasetItem]) -> str:
        stats = cls.compute_stats(items)
        md = f"""# 🚀 BuffData Optimization Audit Report

### 📊 Dataset Overview
- **Total Records**: `{stats['total_items']}`
- **Scored Records**: `{stats['scored_items']}`
- **Average Quality Score**: `{stats['mean_score']:.2f} / 10.0`
- **Median Quality Score**: `{stats['median_score']:.2f} / 10.0`
- **Score Range**: `[{stats['min_score']:.2f} - {stats['max_score']:.2f}]` (Std: `{stats['std_score']:.2f}`)

### 🎯 Quality Breakdown by Dimension
| Metric | Average Score |
| :--- | :--- |
| 📖 **Clarity & Fluency** | `{stats['mean_clarity']:.2f} / 10.0` |
| 🎯 **Factual Accuracy** | `{stats['mean_accuracy']:.2f} / 10.0` |
| 🧠 **Reasoning Depth** | `{stats['mean_depth']:.2f} / 10.0` |
| 📋 **Instruction Adherence** | `{stats['mean_instruction']:.2f} / 10.0` |

### ⚠️ Top Identified Defects & Issues
"""
        if stats["top_issues"]:
            for issue, count in stats["top_issues"]:
                md += f"- **{issue}**: `{count}` occurrences\n"
        else:
            md += "- *No significant defects reported.*\n"

        return md

    @classmethod
    def generate_html_report(cls, items: List[DatasetItem], output_file: Path, provider: str = "configured provider") -> Path:
        provider = escape(provider, quote=True)
        stats = cls.compute_stats(items)
        sample_rows = items[:10]

        samples_html = ""
        for i, item in enumerate(sample_rows):
            p, r = item.get_prompt_and_response()
            p, r = escape(p, quote=True), escape(r, quote=True)
            score_badge = ""
            if item.quality_score:
                score = item.quality_score.overall_score
                color = "#10b981" if score >= 8.0 else ("#f59e0b" if score >= 6.0 else "#ef4444")
                score_badge = f'<span class="badge" style="background:{color}; color:white;">Score: {score:.1f}/10</span>'

            samples_html += f"""
            <div class="sample-card">
                <div class="sample-header">
                    <strong>Sample #{i+1} (ID: {escape(str(item.id), quote=True)})</strong>
                    {score_badge}
                </div>
                <div class="prompt-box">
                    <div class="box-title">Instruction / Prompt:</div>
                    <pre>{p}</pre>
                </div>
                <div class="response-box">
                    <div class="box-title">Response:</div>
                    <pre>{r}</pre>
                </div>
            </div>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BuffData Optimization Report</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #6366f1;
            --accent: #38bdf8;
            --border: #334155;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 2.2rem;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
        }}
        .subtitle {{
            color: var(--text-muted);
            margin-bottom: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
        }}
        .metric-title {{
            font-size: 0.9rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .metric-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #fff;
            margin-top: 8px;
        }}
        .section-title {{
            font-size: 1.4rem;
            margin-top: 40px;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
        }}
        .sample-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .sample-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .badge {{
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .box-title {{
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--accent);
            margin-bottom: 6px;
        }}
        pre {{
            background: #090d16;
            padding: 12px;
            border-radius: 8px;
            overflow-x: auto;
            white-space: pre-wrap;
            color: #e2e8f0;
            font-size: 0.9rem;
            margin: 0 0 15px 0;
            border: 1px solid #1e293b;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 BuffData Optimization Report</h1>
        <div class="subtitle">Audited with {provider}</div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">Total Records</div>
                <div class="metric-value">{stats['total_items']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Mean Quality Score</div>
                <div class="metric-value" style="color:#10b981;">{stats['mean_score']:.2f} <span style="font-size:1rem;">/ 10</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Reasoning Depth</div>
                <div class="metric-value" style="color:#6366f1;">{stats['mean_depth']:.2f} <span style="font-size:1rem;">/ 10</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Factual Accuracy</div>
                <div class="metric-value" style="color:#38bdf8;">{stats['mean_accuracy']:.2f} <span style="font-size:1rem;">/ 10</span></div>
            </div>
        </div>

        <div class="section-title">🔍 Inspected Samples</div>
        {samples_html}
    </div>
</body>
</html>
"""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html, encoding="utf-8")
        return output_file
