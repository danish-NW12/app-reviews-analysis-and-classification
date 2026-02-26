"""Dashboard data service: runs analysis pipeline and shapes response for frontend."""

import json
import os
from pathlib import Path

# Project root (parent of api/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _build_genai_insights(
    kpis: dict,
    top_issues: list,
    top_positive: list,
    sentiment_dist: list,
    severity_dist: list,
    weekly_neg: list,
    weekly_pos: list,
    anomalies: list,
    churn_dist: list,
    version_breakdown: list,
    emerging_themes: list,
) -> dict:
    """
    Call OpenAI to generate extensive insights: overview, findings, suggestions, and structured action items.
    Returns { overview: "", findings: [], suggestions: [], actions: [] } or empty dict if OPENAI_API_KEY is not set or call fails.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return {"overview": "", "findings": [], "suggestions": [], "actions": []}
    try:
        from openai import OpenAI
        client = OpenAI()

        # Rich dataset summary for the model
        top_issue_names = [x.get("category", "") for x in (top_issues or [])[:5]]
        top_positive_names = [x.get("category", "") for x in (top_positive or [])[:5]]
        sentiment_summary = {x.get("name"): {"count": x.get("value"), "pct": x.get("percentage")} for x in (sentiment_dist or [])}
        severity_summary = {x.get("severity"): x.get("count") for x in (severity_dist or [])}
        churn_summary = {x.get("risk", ""): x.get("count") for x in (churn_dist or [])}
        version_top = [{"version": x.get("version"), "count": x.get("count")} for x in (version_breakdown or [])[:5]]
        theme_labels = [x.get("label", "") for x in (emerging_themes or [])[:10]]
        anomaly_sample = [
            {"date": a.get("date"), "type": a.get("type"), "value": a.get("value"), "severity": a.get("severity")}
            for a in (anomalies or [])[:5]
        ]

        summary = {
            "dataset": {
                "total_reviews": kpis.get("totalReviews"),
                "average_rating": kpis.get("averageRating"),
                "sentiment_f1_score": kpis.get("sentimentF1"),
            },
            "sentiment": {
                "negative_pct": kpis.get("negativePercentage"),
                "positive_pct": kpis.get("positivePercentage"),
                "breakdown": sentiment_summary,
            },
            "issues_and_severity": {
                "critical_severity_count": kpis.get("criticalSeverity"),
                "top_5_issue_categories": top_issue_names,
                "severity_distribution": severity_summary,
            },
            "positive_side": {
                "positive_pct": kpis.get("positivePercentage"),
                "high_rating_count_4_5_stars": kpis.get("highRatingCount"),
                "low_churn_count_happy_users": kpis.get("lowChurnCount"),
                "what_users_praise_top_categories": top_positive_names,
            },
            "churn": {
                "high_risk_pct": kpis.get("churnRisk"),
                "distribution": churn_summary,
            },
            "trends": {
                "weekly_negative_sample": [{"week": x.get("week"), "pct": x.get("percentage")} for x in (weekly_neg or [])[:6]],
                "weekly_positive_sample": [{"week": x.get("week"), "pct": x.get("percentage")} for x in (weekly_pos or [])[:6]],
                "anomaly_count": len(anomalies or []),
                "anomaly_examples": anomaly_sample,
            },
            "operations": {
                "bug_resolution_lag_days": kpis.get("bugResolutionLagDays"),
                "version_breakdown_top": version_top,
            },
            "emerging_themes_from_clustering": theme_labels,
        }

        system_prompt = (
            "You are an expert product and analytics advisor for app teams. You analyze review analytics and produce "
            "structured, actionable insights. You always respond with valid JSON only: no markdown, no code fences, no extra text."
        )
        user_prompt = (
            "Below is a comprehensive app review analytics summary. Produce a structured insight report.\n\n"
            "**Data summary:**\n" + json.dumps(summary, indent=2) + "\n\n"
            "**Required output format (strict JSON object):**\n"
            '{\n'
            '  "overview": "2-4 sentences summarizing the dataset and overall health: volume, sentiment balance, main strengths and main concerns.",\n'
            '  "findings": [\n'
            '    "First key finding with specific numbers (e.g. X% negative, Y critical issues).",\n'
            '    "Second finding (e.g. churn risk, rating trend).",\n'
            '    "Third finding (e.g. what users praise, severity mix).",\n'
            '    "Fourth finding (e.g. weekly trends or anomalies).",\n'
            '    "Fifth finding if relevant (e.g. version or resolution lag)."\n'
            "  ],\n"
            '  "suggestions": [\n'
            '    "First detailed suggestion: what to do, why, and expected impact (1-2 sentences).",\n'
            '    "Second suggestion with brief rationale.",\n'
            '    "Third suggestion.",\n'
            '    "Fourth suggestion.",\n'
            '    "Fifth suggestion."\n'
            "  ],\n"
            '  "actions": [\n'
            '    {"title": "Short action title", "description": "What to do and why (1 sentence).", "priority": "P0", "type": "fix"},\n'
            '    {"title": "...", "description": "...", "priority": "P1", "type": "improve"},\n'
            '    {"title": "...", "description": "...", "priority": "P2", "type": "monitor"},\n'
            '    {"title": "...", "description": "...", "priority": null, "type": "celebrate"}\n'
            "  ]\n"
            "}\n\n"
            "Rules: Use the exact keys overview, findings, suggestions, actions. findings and suggestions must be arrays of strings. "
            "actions must be an array of 3 to 6 objects. Each action has: title (string), description (string), optional priority (\"P0\"|\"P1\"|\"P2\"), "
            "optional type: \"fix\"|\"improve\"|\"monitor\" for improvement actions, \"celebrate\"|\"promote\" for positive/good-side actions. "
            "Each finding should cite specific metrics. Each suggestion should be actionable. "
            "Include both improvement actions (fix/improve/monitor) and positive actions (celebrate/promote) where the data supports it."
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1600,
        )
        content = (resp.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        data = json.loads(content)
        if isinstance(data, dict) and "overview" in data:
            raw_actions = data.get("actions") or []
            actions = []
            for a in raw_actions[:10]:
                if isinstance(a, dict) and (a.get("title") or a.get("description")):
                    actions.append({
                        "title": str(a.get("title", "")),
                        "description": str(a.get("description", "")),
                        "priority": a.get("priority") if a.get("priority") in ("P0", "P1", "P2") else None,
                        "type": a.get("type") if a.get("type") in ("fix", "improve", "monitor", "celebrate", "promote") else None,
                    })
            return {
                "overview": str(data.get("overview", "")).strip() or "",
                "findings": [str(s) for s in (data.get("findings") or [])[:6] if s],
                "suggestions": [str(s) for s in (data.get("suggestions") or [])[:6] if s],
                "actions": actions,
            }
    except Exception:
        pass
    return {"overview": "", "findings": [], "suggestions": [], "actions": []}


def _ensure_project_in_path() -> None:
    import sys
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def build_dashboard_payload_from_results(df: "pd.DataFrame", results: dict) -> dict:
    """
    Build dashboard payload (camelCase) from pipeline outputs df and results.
    Used when serving from DB or after run_full_pipeline.
    """
    import pandas as pd  # noqa: F811

    kpis = results.get("dashboard_kpis", {})
    trends = results.get("trends", {})

    # --- KPIs ---
    total = len(df)
    sentiment_metrics = results.get("sentiment_metrics", {})
    sentiment_f1 = sentiment_metrics.get("f1_macro") if isinstance(sentiment_metrics, dict) else None
    avg_rating = None
    if "score" in df.columns and df["score"].notna().any():
        avg_rating = float(df["score"].mean())

    churn_dist = kpis.get("churn_risk_distribution") or {}
    high_risk = churn_dist.get("High", 0)
    churn_pct = round(100 * high_risk / total, 1) if total else None

    bug_lag = kpis.get("bug_resolution_lag_days")
    kpis_payload = {
        "negativePercentage": round(kpis.get("pct_negative_reviews", 0), 1),
        "positivePercentage": round(kpis.get("pct_positive_reviews", 0), 1),
        "criticalSeverity": kpis.get("critical_severity_count", 0),
        "totalReviews": total,
        "sentimentF1": round(sentiment_f1, 2) if sentiment_f1 is not None else None,
        "averageRating": round(avg_rating, 1) if avg_rating is not None else None,
        "churnRisk": churn_pct,
        "bugResolutionLagDays": bug_lag,
        "highRatingCount": kpis.get("high_rating_count", 0),
        "lowChurnCount": kpis.get("low_churn_count", 0),
    }

    # --- Sentiment distribution (order: Positive, Neutral, Negative) ---
    sentiment_order = ["positive", "neutral", "negative"]
    sentiment_names = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
    sentiment_dist: list[dict] = []
    if "sentiment" in df.columns:
        vc = df["sentiment"].astype(str).str.lower().value_counts()
        for key in sentiment_order:
            cnt = int(vc.get(key, 0))
            pct = round(100 * cnt / total, 1) if total else 0
            sentiment_dist.append({
                "name": sentiment_names[key],
                "value": cnt,
                "percentage": pct,
            })

    # --- Severity distribution (buckets) ---
    severity_buckets = [
        ("Critical", 0.7, 1.01, "#ef4444"),
        ("High", 0.5, 0.7, "#f97316"),
        ("Medium", 0.3, 0.5, "#f59e0b"),
        ("Low", 0.0, 0.3, "#84cc16"),
    ]
    severity_dist: list[dict] = []
    if "severity_score" in df.columns:
        s = df["severity_score"].dropna()
        for label, low, high, color in severity_buckets:
            cnt = int(((s >= low) & (s < high)).sum())
            severity_dist.append({"severity": label, "count": cnt, "color": color})

    # --- Top issue categories ---
    top5 = kpis.get("top_5_issue_categories") or {}
    top_list = [
        {"category": k, "count": v, "percentage": round(100 * v / total, 1) if total else 0, "trend": "stable"}
        for k, v in list(top5.items())[:5]
    ]

    # --- Churn risk distribution (frontend: "High Risk", "Medium Risk", "Low Risk") ---
    risk_map = {"High": "High Risk", "Medium": "Medium Risk", "Low": "Low Risk"}
    churn_list = []
    for tier in ("High", "Medium", "Low"):
        count = churn_dist.get(tier, 0)
        label = risk_map[tier]
        pct = round(100 * count / total, 1) if total else 0
        churn_list.append({"risk": label, "count": int(count), "percentage": pct})

    # --- App rating trend ---
    rating_trend_raw = kpis.get("app_rating_trend") or {}
    rating_list = [
        {"week": str(k), "rating": round(float(v), 1), "reviews": 0}
        for k, v in list(rating_trend_raw.items())[:8]
    ]

    # --- Version breakdown ---
    version_raw = kpis.get("version_wise_issue_breakdown") or {}
    version_list = [
        {"version": f"v{k}" if not str(k).startswith("v") else str(k), "count": v, "percentage": round(100 * v / total, 1) if total else 0}
        for k, v in list(version_raw.items())[:10]
    ]

    # --- Weekly negative trend ---
    weekly_neg_raw = kpis.get("weekly_negative_trend") or {}
    weekly_neg_list = [
        {"week": str(k), "percentage": round(float(v), 1), "count": 0}
        for k, v in list(weekly_neg_raw.items())[:8]
    ]

    # --- Weekly positive trend (best side) ---
    weekly_pos_raw = kpis.get("weekly_positive_trend") or {}
    weekly_pos_list = [
        {"week": str(k), "percentage": round(float(v), 1), "count": 0}
        for k, v in list(weekly_pos_raw.items())[:8]
    ]

    # --- Top positive categories (what users praise) ---
    top_pos = kpis.get("top_5_positive_categories") or {}
    top_positive_list = [
        {"category": k, "count": v, "percentage": round(100 * v / total, 1) if total else 0, "trend": "stable"}
        for k, v in list(top_pos.items())[:5]
    ]

    # --- Emerging themes (clustering) ---
    clustering = results.get("clustering", {}) or {}
    cluster_themes = clustering.get("cluster_themes", {})
    emerging_themes = [{"id": k, "label": v} for k, v in cluster_themes.items()] if isinstance(cluster_themes, dict) else []

    # --- Trend anomalies ---
    anomalies_raw = trends.get("anomalies", [])[:20]
    anomaly_list: list[dict] = []
    for a in anomalies_raw:
        if not isinstance(a, dict):
            continue
        date_val = a.get("_date", a.get("date", ""))
        z = a.get("zscore", 0)
        try:
            z = float(z)
        except (TypeError, ValueError):
            z = 0
        anomaly_list.append({
            "date": str(date_val),
            "type": "Spike" if z > 0 else "Drop",
            "metric": "Review count",
            "value": f"{'+' if z > 0 else ''}{z:.1f}σ",
            "severity": "Critical" if abs(z) >= 3 else ("High" if abs(z) >= 2.5 else "Medium"),
        })

    # --- GenAI insights (optional; requires OPENAI_API_KEY when running pipeline) ---
    gen_ai_insights = _build_genai_insights(
        kpis_payload,
        top_list,
        top_positive_list,
        sentiment_dist,
        severity_dist,
        weekly_neg_list,
        weekly_pos_list,
        anomaly_list,
        churn_list,
        version_list,
        emerging_themes,
    )

    return {
        "kpis": kpis_payload,
        "sentimentDistribution": sentiment_dist,
        "severityDistribution": severity_dist,
        "topIssueCategories": top_list,
        "topPositiveCategories": top_positive_list,
        "churnRiskDistribution": churn_list,
        "appRatingTrend": rating_list,
        "versionBreakdown": version_list,
        "weeklyNegativeReviews": weekly_neg_list,
        "weeklyPositiveReviews": weekly_pos_list,
        "trendAnomalies": anomaly_list,
        "emergingThemes": emerging_themes,
        "genAiInsights": gen_ai_insights,
    }


def get_dashboard_data(output_dir: Path | None = None, csv_names: list | None = None) -> dict:
    """
    Run full analysis pipeline and return dashboard payload as a dict
    compatible with DashboardResponse (camelCase for frontend).
    """
    _ensure_project_in_path()
    from main.analysis_pipeline import run_full_pipeline

    out_dir = output_dir or PROJECT_ROOT / "output"
    if not out_dir.exists():
        return _empty_dashboard("Output directory not found. Run conversion and ensure output/ exists.")

    try:
        df, results = run_full_pipeline(output_dir=out_dir, csv_names=csv_names)
    except FileNotFoundError as e:
        return _empty_dashboard(str(e))
    except Exception as e:
        return _empty_dashboard(f"Pipeline error: {e}")

    return build_dashboard_payload_from_results(df, results)


def _empty_dashboard(message: str = "No data available.") -> dict:
    """Return minimal dashboard payload when pipeline fails or no data."""
    return {
        "kpis": {
            "negativePercentage": 0,
            "positivePercentage": 0,
            "criticalSeverity": 0,
            "totalReviews": 0,
            "sentimentF1": None,
            "averageRating": None,
            "churnRisk": None,
            "bugResolutionLagDays": None,
            "highRatingCount": 0,
            "lowChurnCount": 0,
        },
        "sentimentDistribution": [],
        "severityDistribution": [],
        "topIssueCategories": [],
        "topPositiveCategories": [],
        "churnRiskDistribution": [],
        "appRatingTrend": [],
        "versionBreakdown": [],
        "weeklyNegativeReviews": [],
        "weeklyPositiveReviews": [],
        "trendAnomalies": [],
        "emergingThemes": [],
        "genAiInsights": {"overview": "", "findings": [], "suggestions": [], "actions": []},
        "error": message,
    }
