"""Pydantic schemas for dashboard API responses."""

from pydantic import BaseModel, Field, field_validator


class KPIsSchema(BaseModel):
    """Dashboard KPIs (camelCase for frontend)."""

    negative_percentage: float = Field(..., alias="negativePercentage")
    critical_severity: int = Field(..., alias="criticalSeverity")
    total_reviews: int = Field(..., alias="totalReviews")
    sentiment_f1: float | None = Field(None, alias="sentimentF1")
    average_rating: float | None = Field(None, alias="averageRating")
    churn_risk: float | None = Field(None, alias="churnRisk")
    bug_resolution_lag_days: int | None = Field(None, alias="bugResolutionLagDays")
    positive_percentage: float = Field(0, alias="positivePercentage")
    high_rating_count: int = Field(0, alias="highRatingCount")
    low_churn_count: int = Field(0, alias="lowChurnCount")

    model_config = {"populate_by_name": True}


class EmergingThemeItem(BaseModel):
    """Cluster theme from LLM labeling."""

    id: str
    label: str


class SentimentDistributionItem(BaseModel):
    """Single sentiment bucket for pie chart."""

    name: str
    value: int
    percentage: float


class SeverityDistributionItem(BaseModel):
    """Single severity bucket for bar chart."""

    severity: str
    count: int
    color: str


class TopIssueCategoryItem(BaseModel):
    """Top issue category row."""

    category: str
    count: int
    percentage: float
    trend: str = "stable"  # up | down | stable


class ChurnRiskItem(BaseModel):
    """Churn risk tier for chart."""

    risk: str  # "High Risk" | "Medium Risk" | "Low Risk"
    count: int
    percentage: float


class RatingTrendItem(BaseModel):
    """Weekly rating trend point."""

    week: str
    rating: float
    reviews: int = 0


class VersionBreakdownItem(BaseModel):
    """Version-wise review count."""

    version: str
    count: int
    percentage: float


class WeeklyNegativeItem(BaseModel):
    """Weekly negative percentage point."""

    week: str
    percentage: float
    count: int = 0


class TrendAnomalyItem(BaseModel):
    """Trend anomaly row for table."""

    date: str
    type: str  # Spike | Drop
    metric: str
    value: str
    severity: str  # Critical | High | Medium | Low


class ActionItem(BaseModel):
    """Single action item from GenAI: title, description, optional priority and type."""

    title: str = ""
    description: str = ""
    priority: str | None = None  # P0 | P1 | P2
    action_type: str | None = Field(None, alias="type")  # fix | improve | monitor | celebrate | promote

    model_config = {"populate_by_name": True}


class GenAIInsightsSchema(BaseModel):
    """Structured GenAI-generated insights: overview, findings, suggestions, actions."""

    overview: str = ""
    findings: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    actions: list[ActionItem] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class DashboardResponse(BaseModel):
    """Full dashboard payload for frontend."""

    kpis: KPIsSchema
    sentiment_distribution: list[SentimentDistributionItem] = Field(
        default_factory=list, alias="sentimentDistribution"
    )
    severity_distribution: list[SeverityDistributionItem] = Field(
        default_factory=list, alias="severityDistribution"
    )
    top_issue_categories: list[TopIssueCategoryItem] = Field(
        default_factory=list, alias="topIssueCategories"
    )
    churn_risk_distribution: list[ChurnRiskItem] = Field(
        default_factory=list, alias="churnRiskDistribution"
    )
    app_rating_trend: list[RatingTrendItem] = Field(
        default_factory=list, alias="appRatingTrend"
    )
    version_breakdown: list[VersionBreakdownItem] = Field(
        default_factory=list, alias="versionBreakdown"
    )
    weekly_negative_reviews: list[WeeklyNegativeItem] = Field(
        default_factory=list, alias="weeklyNegativeReviews"
    )
    weekly_positive_reviews: list[WeeklyNegativeItem] = Field(
        default_factory=list, alias="weeklyPositiveReviews"
    )
    top_positive_categories: list[TopIssueCategoryItem] = Field(
        default_factory=list, alias="topPositiveCategories"
    )
    trend_anomalies: list[TrendAnomalyItem] = Field(
        default_factory=list, alias="trendAnomalies"
    )
    emerging_themes: list[EmergingThemeItem] = Field(
        default_factory=list, alias="emergingThemes"
    )
    gen_ai_insights: GenAIInsightsSchema = Field(
        default_factory=GenAIInsightsSchema, alias="genAiInsights"
    )

    @field_validator("gen_ai_insights", mode="before")
    @classmethod
    def normalize_gen_ai_insights(cls, v):
        """Accept legacy list[str] from stored payloads; normalize to GenAIInsightsSchema."""
        if isinstance(v, list):
            return GenAIInsightsSchema(overview="", findings=[str(x) for x in v if x], suggestions=[], actions=[])
        if isinstance(v, dict) and v and not isinstance(v, GenAIInsightsSchema):
            raw_actions = v.get("actions", []) or []
            actions = []
            for a in raw_actions[:10]:
                if isinstance(a, dict) and (a.get("title") or a.get("description")):
                    actions.append(ActionItem(
                        title=str(a.get("title", "")),
                        description=str(a.get("description", "")),
                        priority=a.get("priority"),
                        action_type=a.get("type"),
                    ))
            return GenAIInsightsSchema(
                overview=v.get("overview", "") or "",
                findings=v.get("findings", []) or [],
                suggestions=v.get("suggestions", []) or [],
                actions=actions,
            )
        return v

    model_config = {"populate_by_name": True}
