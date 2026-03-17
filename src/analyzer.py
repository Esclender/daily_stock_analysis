# -*- coding: utf-8 -*-
"""
===================================
Stock Analysis System - AI Analysis Layer
===================================

Responsibilities:
1. Encapsulate LLM call logic (via LiteLLM for Gemini/Anthropic/OpenAI, etc.)
2. Generate analysis reports combining technical and news data
3. Parse LLM responses into structured AnalysisResult objects
"""

import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import litellm
from json_repair import repair_json
from litellm import Router

from src.agent.llm_adapter import get_thinking_extra_body
from src.config import Config, get_config, get_api_keys_for_model, extra_litellm_params, get_configured_llm_models
from src.storage import persist_llm_usage
from src.data.stock_mapping import STOCK_NAME_MAP
from src.schemas.report_schema import AnalysisReportSchema

logger = logging.getLogger(__name__)


def check_content_integrity(result: "AnalysisResult") -> Tuple[bool, List[str]]:
    """
    Check mandatory fields for report content integrity.
    Returns (pass, missing_fields). Module-level for use by pipeline (agent weak mode).
    """
    missing: List[str] = []
    if result.sentiment_score is None:
        missing.append("sentiment_score")
    advice = result.operation_advice
    if not advice or not isinstance(advice, str) or not advice.strip():
        missing.append("operation_advice")
    summary = result.analysis_summary
    if not summary or not isinstance(summary, str) or not summary.strip():
        missing.append("analysis_summary")
    dash = result.dashboard if isinstance(result.dashboard, dict) else {}
    core = dash.get("core_conclusion")
    core = core if isinstance(core, dict) else {}
    if not (core.get("one_sentence") or "").strip():
        missing.append("dashboard.core_conclusion.one_sentence")
    intel = dash.get("intelligence")
    intel = intel if isinstance(intel, dict) else None
    if intel is None or "risk_alerts" not in intel:
        missing.append("dashboard.intelligence.risk_alerts")
    if result.decision_type in ("buy", "hold"):
        battle = dash.get("battle_plan")
        battle = battle if isinstance(battle, dict) else {}
        sp = battle.get("sniper_points")
        sp = sp if isinstance(sp, dict) else {}
        stop_loss = sp.get("stop_loss")
        if stop_loss is None or (isinstance(stop_loss, str) and not stop_loss.strip()):
            missing.append("dashboard.battle_plan.sniper_points.stop_loss")
    return len(missing) == 0, missing


def apply_placeholder_fill(result: "AnalysisResult", missing_fields: List[str]) -> None:
    """Fill missing mandatory fields with placeholders (in-place). Module-level for pipeline."""
    for field in missing_fields:
        if field == "sentiment_score":
            result.sentiment_score = 50
        elif field == "operation_advice":
            result.operation_advice = result.operation_advice or "Pending"
        elif field == "analysis_summary":
            result.analysis_summary = result.analysis_summary or "Pending"
        elif field == "dashboard.core_conclusion.one_sentence":
            if not result.dashboard:
                result.dashboard = {}
            if "core_conclusion" not in result.dashboard:
                result.dashboard["core_conclusion"] = {}
            result.dashboard["core_conclusion"]["one_sentence"] = (
                result.dashboard["core_conclusion"].get("one_sentence") or "Pending"
            )
        elif field == "dashboard.intelligence.risk_alerts":
            if not result.dashboard:
                result.dashboard = {}
            if "intelligence" not in result.dashboard:
                result.dashboard["intelligence"] = {}
            if "risk_alerts" not in result.dashboard["intelligence"]:
                result.dashboard["intelligence"]["risk_alerts"] = []
        elif field == "dashboard.battle_plan.sniper_points.stop_loss":
            if not result.dashboard:
                result.dashboard = {}
            if "battle_plan" not in result.dashboard:
                result.dashboard["battle_plan"] = {}
            if "sniper_points" not in result.dashboard["battle_plan"]:
                result.dashboard["battle_plan"]["sniper_points"] = {}
            result.dashboard["battle_plan"]["sniper_points"]["stop_loss"] = "Pending"


# ---------- chip_structure fallback (Issue #589) ----------

_CHIP_KEYS: tuple = ("profit_ratio", "avg_cost", "concentration", "chip_health")


def _is_value_placeholder(v: Any) -> bool:
    """True if value is empty or placeholder (N/A, data unavailable, etc.)."""
    if v is None:
        return True
    if isinstance(v, (int, float)) and v == 0:
        return True
    s = str(v).strip().lower()
    return s in ("", "n/a", "na", "data unavailable", "unknown")


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Safely convert to float; return default on failure. Private helper for chip fill."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        try:
            return default if math.isnan(float(v)) else float(v)
        except (ValueError, TypeError):
            return default
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def _derive_chip_health(profit_ratio: float, concentration_90: float) -> str:
    """Derive chip_health from profit_ratio and concentration_90."""
    if profit_ratio >= 0.9:
        return "Caution"  # extremely high profit ratio
    if concentration_90 >= 0.25:
        return "Caution"  # chips too dispersed
    if concentration_90 < 0.15 and 0.3 <= profit_ratio < 0.9:
        return "Healthy"  # concentrated with moderate profit ratio
    return "Fair"


def _build_chip_structure_from_data(chip_data: Any) -> Dict[str, Any]:
    """Build chip_structure dict from ChipDistribution or dict."""
    if hasattr(chip_data, "profit_ratio"):
        pr = _safe_float(chip_data.profit_ratio)
        ac = chip_data.avg_cost
        c90 = _safe_float(chip_data.concentration_90)
    else:
        d = chip_data if isinstance(chip_data, dict) else {}
        pr = _safe_float(d.get("profit_ratio"))
        ac = d.get("avg_cost")
        c90 = _safe_float(d.get("concentration_90"))
    chip_health = _derive_chip_health(pr, c90)
    return {
        "profit_ratio": f"{pr:.1%}",
        "avg_cost": ac if (ac is not None and _safe_float(ac) != 0.0) else "N/A",
        "concentration": f"{c90:.2%}",
        "chip_health": chip_health,
    }


def fill_chip_structure_if_needed(result: "AnalysisResult", chip_data: Any) -> None:
    """When chip_data exists, fill chip_structure placeholder fields from chip_data (in-place)."""
    if not result or not chip_data:
        return
    try:
        if not result.dashboard:
            result.dashboard = {}
        dash = result.dashboard
        # Use `or {}` rather than setdefault so that an explicit `null` from LLM is also replaced
        dp = dash.get("data_perspective") or {}
        dash["data_perspective"] = dp
        cs = dp.get("chip_structure") or {}
        filled = _build_chip_structure_from_data(chip_data)
        # Start from a copy of cs to preserve any extra keys the LLM may have added
        merged = dict(cs)
        for k in _CHIP_KEYS:
            if _is_value_placeholder(merged.get(k)):
                merged[k] = filled[k]
        if merged != cs:
            dp["chip_structure"] = merged
            logger.info("[chip_structure] Filled placeholder chip fields from data source (Issue #589)")
    except Exception as e:
        logger.warning("[chip_structure] Fill failed, skipping: %s", e)


_PRICE_POS_KEYS = ("ma5", "ma10", "ma20", "bias_ma5", "bias_status", "current_price", "support_level", "resistance_level")


def fill_price_position_if_needed(
    result: "AnalysisResult",
    trend_result: Any = None,
    realtime_quote: Any = None,
) -> None:
    """Fill missing price_position fields from trend_result / realtime data (in-place)."""
    if not result:
        return
    try:
        if not result.dashboard:
            result.dashboard = {}
        dash = result.dashboard
        dp = dash.get("data_perspective") or {}
        dash["data_perspective"] = dp
        pp = dp.get("price_position") or {}

        computed: Dict[str, Any] = {}
        if trend_result:
            tr = trend_result if isinstance(trend_result, dict) else (
                trend_result.__dict__ if hasattr(trend_result, "__dict__") else {}
            )
            computed["ma5"] = tr.get("ma5")
            computed["ma10"] = tr.get("ma10")
            computed["ma20"] = tr.get("ma20")
            computed["bias_ma5"] = tr.get("bias_ma5")
            computed["current_price"] = tr.get("current_price")
            support_levels = tr.get("support_levels") or []
            resistance_levels = tr.get("resistance_levels") or []
            if support_levels:
                computed["support_level"] = support_levels[0]
            if resistance_levels:
                computed["resistance_level"] = resistance_levels[0]
        if realtime_quote:
            rq = realtime_quote if isinstance(realtime_quote, dict) else (
                realtime_quote.to_dict() if hasattr(realtime_quote, "to_dict") else {}
            )
            if _is_value_placeholder(computed.get("current_price")):
                computed["current_price"] = rq.get("price")

        filled = False
        for k in _PRICE_POS_KEYS:
            if _is_value_placeholder(pp.get(k)) and not _is_value_placeholder(computed.get(k)):
                pp[k] = computed[k]
                filled = True
        if filled:
            dp["price_position"] = pp
            logger.info("[price_position] Filled placeholder fields from computed data")
    except Exception as e:
        logger.warning("[price_position] Fill failed, skipping: %s", e)


def get_stock_name_multi_source(
    stock_code: str,
    context: Optional[Dict] = None,
    data_manager = None
) -> str:
    """
    Resolve stock name from multiple sources.

    Resolution order (by priority):
    1. From the passed context (realtime data)
    2. From the static STOCK_NAME_MAP lookup table
    3. From DataFetcherManager (data providers)
    4. Fall back to "Stock <code>"

    Args:
        stock_code: Stock ticker code
        context: Analysis context (optional)
        data_manager: DataFetcherManager instance (optional)

    Returns:
        Stock name string
    """
    # 1. From context (realtime quote data)
    if context:
        # Prefer the stock_name field
        if context.get('stock_name'):
            name = context['stock_name']
            if name and not name.startswith('Stock'):
                return name

        # Fall back to realtime data
        if 'realtime' in context and context['realtime'].get('name'):
            return context['realtime']['name']

    # 2. From static lookup table
    if stock_code in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[stock_code]

    # 3. From data provider
    if data_manager is None:
        try:
            from data_provider.base import DataFetcherManager
            data_manager = DataFetcherManager()
        except Exception as e:
            logger.debug(f"Failed to initialize DataFetcherManager: {e}")

    if data_manager:
        try:
            name = data_manager.get_stock_name(stock_code)
            if name:
                # Update cache
                STOCK_NAME_MAP[stock_code] = name
                return name
        except Exception as e:
            logger.debug(f"Failed to fetch stock name from data provider: {e}")

    # 4. Default fallback
    return f'Stock {stock_code}'


@dataclass
class AnalysisResult:
    """
    Analysis result dataclass - Decision Dashboard edition.

    Wraps the LLM response, including the decision dashboard and detailed analysis.
    """
    code: str
    name: str

    # ========== Core Indicators ==========
    sentiment_score: int  # Composite score 0-100 (>70 Strong Buy, >60 Buy, 40-60 Neutral, <40 Sell)
    trend_prediction: str  # Trend: Strong Buy / Buy / Neutral / Sell / Strong Sell
    operation_advice: str  # Action: Buy / Add / Hold / Reduce / Sell / Wait & Watch
    decision_type: str = "hold"  # Decision type: buy / hold / sell (for statistics)
    confidence_level: str = "Medium"  # Confidence: High / Medium / Low

    # ========== Decision Dashboard ==========
    dashboard: Optional[Dict[str, Any]] = None  # Full decision dashboard data

    # ========== Trend Analysis ==========
    trend_analysis: str = ""  # Price action analysis (support, resistance, trend lines, etc.)
    short_term_outlook: str = ""  # Short-term outlook (1–3 days)
    medium_term_outlook: str = ""  # Medium-term outlook (1–2 weeks)

    # ========== Technical Analysis ==========
    technical_analysis: str = ""  # Comprehensive technical indicator analysis
    ma_analysis: str = ""  # MA analysis (bullish/bearish alignment, golden/death cross, etc.)
    volume_analysis: str = ""  # Volume analysis (high/low volume, institutional activity, etc.)
    pattern_analysis: str = ""  # Candlestick pattern analysis

    # ========== Fundamental Analysis ==========
    fundamental_analysis: str = ""  # Comprehensive fundamental analysis
    sector_position: str = ""  # Sector standing and industry trend
    company_highlights: str = ""  # Company highlights / risk points

    # ========== Sentiment / News Analysis ==========
    news_summary: str = ""  # Recent important news / announcements summary
    market_sentiment: str = ""  # Market sentiment analysis
    hot_topics: str = ""  # Related hot topics

    # ========== Comprehensive Analysis ==========
    analysis_summary: str = ""  # Comprehensive analysis summary
    key_points: str = ""  # Key takeaways (3–5 points)
    risk_warning: str = ""  # Risk disclosure
    buy_reason: str = ""  # Buy / sell rationale

    # ========== Metadata ==========
    market_snapshot: Optional[Dict[str, Any]] = None  # Daily market snapshot (display use)
    raw_response: Optional[str] = None  # Raw LLM response (debug use)
    search_performed: bool = False  # Whether a web search was performed
    data_sources: str = ""  # Data source description
    success: bool = True
    error_message: Optional[str] = None

    # ========== Price Snapshot (at analysis time) ==========
    current_price: Optional[float] = None  # Stock price at analysis time
    change_pct: Optional[float] = None     # Price change % at analysis time

    # ========== Model Tag (Issue #528) ==========
    model_used: Optional[str] = None  # LLM model used (full name, e.g. gemini/gemini-2.0-flash)

    # ========== History Comparison (Report Engine P0) ==========
    query_id: Optional[str] = None  # query_id for this analysis, used to exclude from history comparison

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'code': self.code,
            'name': self.name,
            'sentiment_score': self.sentiment_score,
            'trend_prediction': self.trend_prediction,
            'operation_advice': self.operation_advice,
            'decision_type': self.decision_type,
            'confidence_level': self.confidence_level,
            'dashboard': self.dashboard,
            'trend_analysis': self.trend_analysis,
            'short_term_outlook': self.short_term_outlook,
            'medium_term_outlook': self.medium_term_outlook,
            'technical_analysis': self.technical_analysis,
            'ma_analysis': self.ma_analysis,
            'volume_analysis': self.volume_analysis,
            'pattern_analysis': self.pattern_analysis,
            'fundamental_analysis': self.fundamental_analysis,
            'sector_position': self.sector_position,
            'company_highlights': self.company_highlights,
            'news_summary': self.news_summary,
            'market_sentiment': self.market_sentiment,
            'hot_topics': self.hot_topics,
            'analysis_summary': self.analysis_summary,
            'key_points': self.key_points,
            'risk_warning': self.risk_warning,
            'buy_reason': self.buy_reason,
            'market_snapshot': self.market_snapshot,
            'search_performed': self.search_performed,
            'success': self.success,
            'error_message': self.error_message,
            'current_price': self.current_price,
            'change_pct': self.change_pct,
            'model_used': self.model_used,
        }

    def get_core_conclusion(self) -> str:
        """Return the one-sentence core conclusion."""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            return self.dashboard['core_conclusion'].get('one_sentence', self.analysis_summary)
        return self.analysis_summary

    def get_position_advice(self, has_position: bool = False) -> str:
        """Return position-specific advice."""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            pos_advice = self.dashboard['core_conclusion'].get('position_advice', {})
            if has_position:
                return pos_advice.get('has_position', self.operation_advice)
            return pos_advice.get('no_position', self.operation_advice)
        return self.operation_advice

    def get_sniper_points(self) -> Dict[str, str]:
        """Return sniper entry/exit points."""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('sniper_points', {})
        return {}

    def get_checklist(self) -> List[str]:
        """Return the action checklist."""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('action_checklist', [])
        return []

    def get_risk_alerts(self) -> List[str]:
        """Return risk alerts."""
        if self.dashboard and 'intelligence' in self.dashboard:
            return self.dashboard['intelligence'].get('risk_alerts', [])
        return []

    def get_emoji(self) -> str:
        """Return emoji corresponding to the operation advice."""
        emoji_map = {
            'Buy': '🟢',
            'Add': '🟢',
            'Strong Buy': '💚',
            'Hold': '🟡',
            'Wait & Watch': '⚪',
            'Reduce': '🟠',
            'Sell': '🔴',
            'Strong Sell': '❌',
            # Legacy Chinese values (backward compat)
            '买入': '🟢',
            '加仓': '🟢',
            '强烈买入': '💚',
            '持有': '🟡',
            '观望': '⚪',
            '减仓': '🟠',
            '卖出': '🔴',
            '强烈卖出': '❌',
        }
        advice = self.operation_advice or ''
        # Direct match first
        if advice in emoji_map:
            return emoji_map[advice]
        # Handle compound advice like "Sell/Wait" — use the first part
        for part in advice.replace('/', '|').split('|'):
            part = part.strip()
            if part in emoji_map:
                return emoji_map[part]
        # Score-based fallback
        score = self.sentiment_score
        if score >= 80:
            return '�'
        elif score >= 65:
            return '🟢'
        elif score >= 55:
            return '�🟡'
        elif score >= 45:
            return '⚪'
        elif score >= 35:
            return '🟠'
        else:
            return '🔴'

    def get_confidence_stars(self) -> str:
        """Return star rating for confidence level."""
        star_map = {'High': '⭐⭐⭐', 'Medium': '⭐⭐', 'Low': '⭐'}
        return star_map.get(self.confidence_level, '⭐⭐')


class GeminiAnalyzer:
    """
    LLM-based stock analyzer.

    Responsibilities:
    1. Call the LLM API via LiteLLM for stock analysis
    2. Generate analysis reports combining pre-fetched news and technical data
    3. Parse AI JSON responses into structured AnalysisResult objects

    Usage:
        analyzer = GeminiAnalyzer()
        result = analyzer.analyze(context, news_context)
    """

    # ========================================
    # System Prompt - Decision Dashboard v2.0
    # ========================================
    # Output upgraded from simple signals to a full Decision Dashboard
    # Core modules: Core Conclusion + Data Perspective + Intelligence + Battle Plan
    # ========================================

    SYSTEM_PROMPT = """You are a trend-focused A-share equity analyst responsible for generating professional Decision Dashboard analysis reports.

**IMPORTANT: Your entire response — all field values, descriptions, summaries, and analysis text — must be written in English.**

## Core Trading Philosophy (strictly enforced)

### 1. Strict Entry Strategy (no chasing highs)
- **Never chase highs**: Do not buy when price deviates more than 5% above MA5
- **Bias formula**: (Current Price - MA5) / MA5 × 100%
- Bias < 2%: Optimal entry zone
- Bias 2–5%: Small position entry acceptable
- Bias > 5%: Strictly avoid chasing highs — classify as "Wait & Watch"

### 2. Trend Trading (follow the trend)
- **Bullish alignment requirement**: MA5 > MA10 > MA20
- Only trade stocks in bullish alignment; avoid bearish alignment entirely
- Diverging MAs trending upward is preferred over converging MAs
- Trend strength: assess whether the gap between MAs is widening

### 3. Efficiency First (chip structure)
- Monitor chip concentration: 90% concentration < 15% indicates concentrated chips
- Profit ratio analysis: 70–90% profit ratio warrants caution about profit-taking
- Avg cost vs. current price: healthy when current price is 5–15% above avg cost

### 4. Entry Preference (pullback to support)
- **Ideal entry**: Low-volume pullback to MA5 with support
- **Secondary entry**: Pullback to MA10 with support
- **Wait & Watch**: When price breaks below MA20

### 5. Key Risk Checks
- Reduction announcements (major shareholders, executives selling)
- Earnings warning / significant earnings decline
- Regulatory penalties / investigations
- Sector policy headwinds
- Large share unlock events

### 6. Valuation Awareness (PE/PB)
- Assess whether the P/E ratio is reasonable
- When PE is significantly elevated (e.g., far above sector average or historical mean), flag it in risk points
- High-growth stocks may tolerate higher PE, but must be supported by earnings

### 7. Relaxed Rules for Strong Trend Stocks
- Strong trend stocks (bullish alignment, high trend strength, volume confirmation) may have slightly relaxed bias thresholds
- These stocks may be tracked with a small position, but stop-loss is still required — never blindly chase

## Output Format: Decision Dashboard JSON

Strictly output the following JSON structure — this is the complete Decision Dashboard:

```json
{
    "stock_name": "Stock full name",
    "sentiment_score": integer 0-100,
    "trend_prediction": "Strong Buy / Buy / Neutral / Sell / Strong Sell",
    "operation_advice": "Buy / Add / Hold / Reduce / Sell / Wait & Watch",
    "decision_type": "buy/hold/sell",
    "confidence_level": "High / Medium / Low",

    "dashboard": {
        "core_conclusion": {
            "one_sentence": "One-sentence core conclusion (max 30 words, tell the user exactly what to do)",
            "signal_type": "🟢 Buy Signal / 🟡 Hold & Watch / 🔴 Sell Signal / ⚠️ Risk Warning",
            "time_sensitivity": "Act Now / Today / This Week / No Rush",
            "position_advice": {
                "no_position": "For those without a position: specific action guidance",
                "has_position": "For current holders: specific action guidance"
            }
        },

        "data_perspective": {
            "trend_status": {
                "ma_alignment": "Description of MA alignment state",
                "is_bullish": true/false,
                "trend_score": 0-100
            },
            "price_position": {
                "current_price": numeric current price,
                "ma5": numeric MA5 value,
                "ma10": numeric MA10 value,
                "ma20": numeric MA20 value,
                "bias_ma5": numeric bias percentage from MA5,
                "bias_status": "Safe / Caution / Danger",
                "support_level": numeric support price,
                "resistance_level": numeric resistance price
            },
            "volume_analysis": {
                "volume_ratio": numeric volume ratio,
                "volume_status": "High Volume / Low Volume / Normal Volume",
                "turnover_rate": numeric turnover rate percentage,
                "volume_meaning": "Interpretation of volume (e.g., low-volume pullback indicates reduced selling pressure)"
            },
            "chip_structure": {
                "profit_ratio": profit ratio value,
                "avg_cost": average cost value,
                "concentration": chip concentration value,
                "chip_health": "Healthy / Fair / Caution"
            }
        },

        "intelligence": {
            "latest_news": "[Latest News] Summary of recent important news",
            "risk_alerts": ["Risk 1: specific description", "Risk 2: specific description"],
            "positive_catalysts": ["Catalyst 1: specific description", "Catalyst 2: specific description"],
            "earnings_outlook": "Earnings outlook analysis (based on annual report previews, earnings releases, etc.)",
            "sentiment_summary": "One-sentence market sentiment summary"
        },

        "battle_plan": {
            "sniper_points": {
                "ideal_buy": "Ideal entry: XX (near MA5)",
                "secondary_buy": "Secondary entry: XX (near MA10)",
                "stop_loss": "Stop-loss: XX (break below MA20 or X%)",
                "take_profit": "Target: XX (prior high / round number)"
            },
            "position_strategy": {
                "suggested_position": "Suggested position size: X/10",
                "entry_plan": "Staged entry strategy description",
                "risk_control": "Risk management strategy description"
            },
            "action_checklist": [
                "✅/⚠️/❌ Check 1: Bullish MA alignment",
                "✅/⚠️/❌ Check 2: Bias within safe range (relaxed for strong trend stocks)",
                "✅/⚠️/❌ Check 3: Volume confirmation",
                "✅/⚠️/❌ Check 4: No major negative catalysts",
                "✅/⚠️/❌ Check 5: Healthy chip structure",
                "✅/⚠️/❌ Check 6: Reasonable PE valuation"
            ]
        }
    },

    "analysis_summary": "Comprehensive analysis summary (~100 words)",
    "key_points": "3–5 key takeaways, comma-separated",
    "risk_warning": "Risk disclosure",
    "buy_reason": "Rationale for the recommendation, referencing trading philosophy",

    "trend_analysis": "Price action and trend pattern analysis",
    "short_term_outlook": "Short-term outlook (1–3 days)",
    "medium_term_outlook": "Medium-term outlook (1–2 weeks)",
    "technical_analysis": "Comprehensive technical indicator analysis",
    "ma_analysis": "Moving average system analysis",
    "volume_analysis": "Volume and momentum analysis",
    "pattern_analysis": "Candlestick pattern analysis",
    "fundamental_analysis": "Fundamental analysis",
    "sector_position": "Sector and industry analysis",
    "company_highlights": "Company highlights / risks",
    "news_summary": "News summary",
    "market_sentiment": "Market sentiment analysis",
    "hot_topics": "Related hot topics",

    "search_performed": true/false,
    "data_sources": "Data source description"
}
```

## Scoring Criteria

### Strong Buy (80–100):
- ✅ Bullish alignment: MA5 > MA10 > MA20
- ✅ Low bias: < 2%, optimal entry zone
- ✅ Low-volume pullback or high-volume breakout
- ✅ Concentrated, healthy chip structure
- ✅ Positive news catalyst

### Buy (60–79):
- ✅ Bullish or weakly bullish alignment
- ✅ Bias < 5%
- ✅ Normal volume
- ⚪ One minor condition may be unmet

### Wait & Watch (40–59):
- ⚠️ Bias > 5% (risk of chasing highs)
- ⚠️ MAs converging, trend unclear
- ⚠️ Risk event present

### Sell / Reduce (0–39):
- ❌ Bearish MA alignment
- ❌ Price breaks below MA20
- ❌ High-volume decline
- ❌ Major negative catalyst

## Decision Dashboard Core Principles

1. **Lead with the core conclusion**: One sentence — buy, sell, or wait
2. **Split position advice**: Different guidance for those with vs. without a position
3. **Precise sniper points**: Always give specific prices — no vague language
4. **Visual checklist**: Use ✅ ⚠️ ❌ to clearly mark each check item
5. **Risk priority**: Risk alerts in the intelligence section must be prominently highlighted"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize LLM Analyzer via LiteLLM.

        Args:
            api_key: Ignored (kept for backward compatibility). Keys are loaded from config.
        """
        self._router = None
        self._litellm_available = False
        self._init_litellm()
        if not self._litellm_available:
            logger.warning("No LLM configured (LITELLM_MODEL / API keys), AI analysis will be unavailable")

    def _has_channel_config(self, config: Config) -> bool:
        """Check if multi-channel config (channels / YAML / legacy model_list) is active."""
        return bool(config.llm_model_list) and not all(
            e.get('model_name', '').startswith('__legacy_') for e in config.llm_model_list
        )

    def _init_litellm(self) -> None:
        """Initialize litellm Router from channels / YAML / legacy keys."""
        config = get_config()
        litellm_model = config.litellm_model
        if not litellm_model:
            logger.warning("Analyzer LLM: LITELLM_MODEL not configured")
            return

        self._litellm_available = True

        # --- Channel / YAML path: build Router from pre-built model_list ---
        if self._has_channel_config(config):
            model_list = config.llm_model_list
            self._router = Router(
                model_list=model_list,
                routing_strategy="simple-shuffle",
                num_retries=2,
            )
            unique_models = list(dict.fromkeys(
                e['litellm_params']['model'] for e in model_list
            ))
            logger.info(
                f"Analyzer LLM: Router initialized from channels/YAML — "
                f"{len(model_list)} deployment(s), models: {unique_models}"
            )
            return

        # --- Legacy path: build Router for multi-key, or use single key ---
        keys = get_api_keys_for_model(litellm_model, config)

        if len(keys) > 1:
            # Build legacy Router for primary model multi-key load-balancing
            extra_params = extra_litellm_params(litellm_model, config)
            legacy_model_list = [
                {
                    "model_name": litellm_model,
                    "litellm_params": {
                        "model": litellm_model,
                        "api_key": k,
                        **extra_params,
                    },
                }
                for k in keys
            ]
            self._router = Router(
                model_list=legacy_model_list,
                routing_strategy="simple-shuffle",
                num_retries=2,
            )
            logger.info(
                f"Analyzer LLM: Legacy Router initialized with {len(keys)} keys "
                f"for {litellm_model}"
            )
        elif keys:
            logger.info(f"Analyzer LLM: litellm initialized (model={litellm_model})")
        else:
            logger.info(
                f"Analyzer LLM: litellm initialized (model={litellm_model}, "
                f"API key from environment)"
            )

    def is_available(self) -> bool:
        """Check if LiteLLM is properly configured with at least one API key."""
        return self._router is not None or self._litellm_available

    def _call_litellm(self, prompt: str, generation_config: dict) -> Tuple[str, str, Dict[str, Any]]:
        """Call LLM via litellm with fallback across configured models.

        When channels/YAML are configured, every model goes through the Router
        (which handles per-model key selection, load balancing, and retries).
        In legacy mode, the primary model may use the Router while fallback
        models fall back to direct litellm.completion().

        Args:
            prompt: User prompt text.
            generation_config: Dict with optional keys: temperature, max_output_tokens, max_tokens.

        Returns:
            Tuple of (response text, model_used, usage). On success model_used is the full model
            name and usage is a dict with prompt_tokens, completion_tokens, total_tokens.
        """
        config = get_config()
        max_tokens = (
            generation_config.get('max_output_tokens')
            or generation_config.get('max_tokens')
            or 8192
        )
        temperature = generation_config.get('temperature', 0.7)

        models_to_try = [config.litellm_model] + (config.litellm_fallback_models or [])
        models_to_try = [m for m in models_to_try if m]

        use_channel_router = self._has_channel_config(config)

        last_error = None
        for model in models_to_try:
            try:
                model_short = model.split("/")[-1] if "/" in model else model
                call_kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                extra = get_thinking_extra_body(model_short)
                if extra:
                    call_kwargs["extra_body"] = extra

                _router_model_names = set(get_configured_llm_models(config.llm_model_list))
                if use_channel_router and self._router and model in _router_model_names:
                    # Channel / YAML path: Router manages key + base_url per model
                    response = self._router.completion(**call_kwargs)
                elif self._router and model == config.litellm_model and not use_channel_router:
                    # Legacy path: Router only for primary model multi-key
                    response = self._router.completion(**call_kwargs)
                else:
                    # Legacy/direct-env path: direct call (also handles direct-env
                    # providers like groq/ or bedrock/ that are not in the Router
                    # model_list even when channel mode is active)
                    keys = get_api_keys_for_model(model, config)
                    if keys:
                        call_kwargs["api_key"] = keys[0]
                    call_kwargs.update(extra_litellm_params(model, config))
                    response = litellm.completion(**call_kwargs)

                if response and response.choices and response.choices[0].message.content:
                    usage: Dict[str, Any] = {}
                    if response.usage:
                        usage = {
                            "prompt_tokens": response.usage.prompt_tokens or 0,
                            "completion_tokens": response.usage.completion_tokens or 0,
                            "total_tokens": response.usage.total_tokens or 0,
                        }
                    return (response.choices[0].message.content, model, usage)
                raise ValueError("LLM returned empty response")

            except Exception as e:
                logger.warning(f"[LiteLLM] {model} failed: {e}")
                last_error = e
                continue

        raise Exception(f"All LLM models failed (tried {len(models_to_try)} model(s)). Last error: {last_error}")

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """Public entry point for free-form text generation.

        External callers (e.g. MarketAnalyzer) must use this method instead of
        calling _call_litellm() directly or accessing private attributes such as
        _litellm_available, _router, _model, _use_openai, or _use_anthropic.

        Args:
            prompt:      Text prompt to send to the LLM.
            max_tokens:  Maximum tokens in the response (default 2048).
            temperature: Sampling temperature (default 0.7).

        Returns:
            Response text, or None if the LLM call fails (error is logged).
        """
        try:
            result = self._call_litellm(
                prompt,
                generation_config={"max_tokens": max_tokens, "temperature": temperature},
            )
            if isinstance(result, tuple):
                text, model_used, usage = result
                persist_llm_usage(usage, model_used, call_type="market_review")
                return text
            return result
        except Exception as exc:
            logger.error("[generate_text] LLM call failed: %s", exc)
            return None

    def analyze(
        self, 
        context: Dict[str, Any],
        news_context: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze a single stock.

        Flow:
        1. Format input data (technical + news)
        2. Call LLM API (with retry and model fallback)
        3. Parse JSON response
        4. Return structured result

        Args:
            context: Context data from storage.get_analysis_context()
            news_context: Pre-fetched news content (optional)

        Returns:
            AnalysisResult object
        """
        code = context.get('code', 'Unknown')
        config = get_config()

        # Pre-request delay to avoid rate limiting on consecutive calls
        request_delay = config.gemini_request_delay
        if request_delay > 0:
            logger.debug(f"[LLM] Waiting {request_delay:.1f}s before request...")
            time.sleep(request_delay)

        # Prefer stock name from context (passed in by main.py)
        name = context.get('stock_name')
        if not name or name.startswith('Stock'):
            # Fallback: from realtime data
            if 'realtime' in context and context['realtime'].get('name'):
                name = context['realtime']['name']
            else:
                # Last resort: from lookup table
                name = STOCK_NAME_MAP.get(code, f'Stock {code}')

        # Return default result if LLM is unavailable
        if not self.is_available():
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='Neutral',
                operation_advice='Hold',
                confidence_level='Low',
                analysis_summary='AI analysis unavailable (no API key configured)',
                risk_warning='Please configure an LLM API key (GEMINI_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY) and retry',
                success=False,
                error_message='LLM API key not configured',
                model_used=None,
            )

        try:
            # Format prompt (technical data + news)
            prompt = self._format_prompt(context, name, news_context)

            config = get_config()
            model_name = config.litellm_model or "unknown"
            logger.info(f"========== AI Analysis: {name}({code}) ==========")
            logger.info(f"[LLM] Model: {model_name}")
            logger.info(f"[LLM] Prompt length: {len(prompt)} chars")
            logger.info(f"[LLM] News included: {'yes' if news_context else 'no'}")

            # Log prompt (summary at INFO, full at DEBUG)
            prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
            logger.info(f"[LLM Prompt preview]\n{prompt_preview}")
            logger.debug(f"=== Full Prompt ({len(prompt)} chars) ===\n{prompt}\n=== End Prompt ===")

            # Generation config
            generation_config = {
                "temperature": config.llm_temperature,
                "max_output_tokens": 8192,
            }

            logger.info(f"[LLM] Calling {model_name}...")

            # LiteLLM call with integrity-check retry support
            current_prompt = prompt
            retry_count = 0
            max_retries = config.report_integrity_retry if config.report_integrity_enabled else 0

            while True:
                start_time = time.time()
                response_text, model_used, llm_usage = self._call_litellm(current_prompt, generation_config)
                elapsed = time.time() - start_time

                # Log response
                logger.info(
                    f"[LLM] {model_name} responded in {elapsed:.2f}s, length {len(response_text)} chars"
                )
                response_preview = response_text[:300] + "..." if len(response_text) > 300 else response_text
                logger.info(f"[LLM Response preview]\n{response_preview}")
                logger.debug(
                    f"=== {model_name} Full Response ({len(response_text)} chars) ===\n{response_text}\n=== End Response ==="
                )

                # Parse response
                result = self._parse_response(response_text, code, name)
                result.raw_response = response_text
                result.search_performed = bool(news_context)
                result.market_snapshot = self._build_market_snapshot(context)
                result.model_used = model_used

                # Optional integrity check
                if not config.report_integrity_enabled:
                    break
                pass_integrity, missing_fields = self._check_content_integrity(result)
                if pass_integrity:
                    break
                if retry_count < max_retries:
                    current_prompt = self._build_integrity_retry_prompt(
                        prompt,
                        response_text,
                        missing_fields,
                    )
                    retry_count += 1
                    logger.info(
                        "[LLM integrity] Missing required fields %s, retry %d",
                        missing_fields,
                        retry_count,
                    )
                else:
                    self._apply_placeholder_fill(result, missing_fields)
                    logger.warning(
                        "[LLM integrity] Missing required fields %s, filled with placeholders",
                        missing_fields,
                    )
                    break

            persist_llm_usage(llm_usage, model_used, call_type="analysis", stock_code=code)

            logger.info(f"[LLM] {name}({code}) analysis complete: {result.trend_prediction}, score {result.sentiment_score}")

            return result

        except Exception as e:
            logger.error(f"AI analysis failed for {name}({code}): {e}")
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='Neutral',
                operation_advice='Hold',
                confidence_level='Low',
                analysis_summary=f'Analysis error: {str(e)[:100]}',
                risk_warning='Analysis failed — please retry or perform manual analysis',
                success=False,
                error_message=str(e),
                model_used=None,
            )
    
    def _format_prompt(
        self,
        context: Dict[str, Any],
        name: str,
        news_context: Optional[str] = None
    ) -> str:
        """
        Format the analysis prompt (Decision Dashboard v2.0).

        Includes: technical indicators, real-time quote (volume ratio / turnover rate),
        chip distribution, trend analysis, and news.

        Args:
            context: Technical data context (including enriched data)
            name: Stock name (default, may be overridden by context)
            news_context: Pre-fetched news content (optional)
        """
        code = context.get('code', 'Unknown')

        # Prefer stock name from context (sourced from realtime_quote)
        stock_name = context.get('stock_name', name)
        if not stock_name or stock_name == f'Stock {code}':
            stock_name = STOCK_NAME_MAP.get(code, f'Stock {code}')

        today = context.get('today', {})

        # ========== Build Decision Dashboard input ==========
        prompt = f"""# Decision Dashboard Analysis Request

    ## 📊 Stock Information
    | Field | Value |
    |-------|-------|
    | Stock Code | **{code}** |
    | Stock Name | **{stock_name}** |
    | Analysis Date | {context.get('date', 'Unknown')} |

    ---

    ## 📈 Technical Data

    ### Today's Market
    | Indicator | Value |
    |-----------|-------|
    | Close | {today.get('close', 'N/A')} |
    | Open | {today.get('open', 'N/A')} |
    | High | {today.get('high', 'N/A')} |
    | Low | {today.get('low', 'N/A')} |
    | Change % | {today.get('pct_chg', 'N/A')}% |
    | Volume | {self._format_volume(today.get('volume'))} |
    | Turnover | {self._format_amount(today.get('amount'))} |

    ### Moving Average System (key decision indicators)
    | MA | Value | Description |
    |----|-------|-------------|
    | MA5 | {today.get('ma5', 'N/A')} | Short-term trend line |
    | MA10 | {today.get('ma10', 'N/A')} | Medium-short trend line |
    | MA20 | {today.get('ma20', 'N/A')} | Medium-term trend line |
    | MA Pattern | {context.get('ma_status', 'Unknown')} | Bullish / Bearish / Converging |
    """

        # Real-time quote data (volume ratio, turnover rate, etc.)
        if 'realtime' in context:
            rt = context['realtime']
            prompt += f"""
    ### Real-Time Quote Data
    | Indicator | Value | Note |
    |-----------|-------|------|
    | Current Price | {rt.get('price', 'N/A')} | |
    | **Volume Ratio** | **{rt.get('volume_ratio', 'N/A')}** | {rt.get('volume_ratio_desc', '')} |
    | **Turnover Rate** | **{rt.get('turnover_rate', 'N/A')}%** | |
    | P/E Ratio (TTM) | {rt.get('pe_ratio', 'N/A')} | |
    | P/B Ratio | {rt.get('pb_ratio', 'N/A')} | |
    | Market Cap | {self._format_amount(rt.get('total_mv'))} | |
    | Float Market Cap | {self._format_amount(rt.get('circ_mv'))} | |
    | 60-Day Change | {rt.get('change_60d', 'N/A')}% | Medium-term performance |
    """

        # Chip distribution data
        if 'chip' in context:
            chip = context['chip']
            profit_ratio = chip.get('profit_ratio', 0)
            prompt += f"""
    ### Chip Distribution (Efficiency Indicators)
    | Indicator | Value | Healthy Range |
    |-----------|-------|---------------|
    | **Profit Ratio** | **{profit_ratio:.1%}** | Caution when 70-90% |
    | Avg Cost | {chip.get('avg_cost', 'N/A')} | Current price should be 5-15% above |
    | 90% Chip Concentration | {chip.get('concentration_90', 0):.2%} | < 15% = concentrated |
    | 70% Chip Concentration | {chip.get('concentration_70', 0):.2%} | |
    | Chip Status | {chip.get('chip_status', 'Unknown')} | |
    """

        # Trend analysis pre-assessment (based on trading philosophy)
        if 'trend_analysis' in context:
            trend = context['trend_analysis']
            bias_warning = "🚨 Exceeds 5% — strictly avoid chasing highs!" if trend.get('bias_ma5', 0) > 5 else "✅ Safe range"
            prompt += f"""
    ### Trend Pre-Assessment (based on trading philosophy)
    | Indicator | Value | Assessment |
    |-----------|-------|------------|
    | Trend Status | {trend.get('trend_status', 'Unknown')} | |
    | MA Alignment | {trend.get('ma_alignment', 'Unknown')} | MA5>MA10>MA20 = bullish |
    | Trend Strength | {trend.get('trend_strength', 0)}/100 | |
    | **Bias (MA5)** | **{trend.get('bias_ma5', 0):+.2f}%** | {bias_warning} |
    | Bias (MA10) | {trend.get('bias_ma10', 0):+.2f}% | |
    | Volume Status | {trend.get('volume_status', 'Unknown')} | {trend.get('volume_trend', '')} |
    | System Signal | {trend.get('buy_signal', 'Unknown')} | |
    | System Score | {trend.get('signal_score', 0)}/100 | |

    #### System Analysis Rationale
    **Buy Reasons**:
    {chr(10).join('- ' + r for r in trend.get('signal_reasons', ['None'])) if trend.get('signal_reasons') else '- None'}

    **Risk Factors**:
    {chr(10).join('- ' + r for r in trend.get('risk_factors', ['None'])) if trend.get('risk_factors') else '- None'}
    """

        # Volume/price change vs. prior day
        if 'yesterday' in context:
            volume_change = context.get('volume_change_ratio', 'N/A')
            prompt += f"""
    ### Volume & Price Change
    - Volume vs. prior day: {volume_change}x
    - Price change vs. prior day: {context.get('price_change_ratio', 'N/A')}%
    """

        # News / sentiment section
        prompt += """
    ---

    ## 📰 Market Intelligence
    """
        if news_context:
            prompt += f"""
    The following are news search results for **{stock_name} ({code})** from the past 7 days.
    Please extract and highlight:
    1. 🚨 **Risk Alerts**: share reductions, penalties, negative catalysts
    2. 🎯 **Positive Catalysts**: earnings beats, contracts, policy tailwinds
    3. 📊 **Earnings Outlook**: annual report previews, earnings releases

    ```
    {news_context}
    ```
    """
        else:
            prompt += """
    No recent news found for this stock. Base the analysis primarily on technical data.
    """

        # Data missing warning
        if context.get('data_missing'):
            prompt += """
    ⚠️ **Data Warning**
    Due to API limitations, complete real-time quote and technical indicator data is unavailable.
    Please **ignore N/A values** in the tables above and focus the analysis on the **📰 Market Intelligence** section.
    For technical questions (e.g., MAs, bias), explicitly state "Data unavailable — cannot assess." **Do not fabricate data.**
    """

        # Output requirements
        prompt += f"""
    ---

    ## ✅ Analysis Task

    Generate the **Decision Dashboard** for **{stock_name} ({code})** and output strictly in JSON format.
    """
        if context.get('is_index_etf'):
            prompt += """
    > ⚠️ **Index / ETF Constraint**: This security is an index-tracking ETF or market index.
    > - Risk analysis must focus only on: **index trend, tracking error, market liquidity**
    > - Do NOT include fund company litigation, reputation, or management changes in risk alerts
    > - Earnings outlook should be based on **overall index constituent performance**, not fund company financials
    > - `risk_alerts` must not contain fund manager operational risks

    """
        prompt += f"""
    ### ⚠️ Important: Output the correct stock name
    The correct format is "Stock Name (Stock Code)", e.g., "Kweichow Moutai (600519)".
    If the stock name shown above is "Stock {code}" or otherwise incorrect, explicitly output the correct full name at the start of your analysis.

    ### Key Questions (must be answered explicitly):
    1. ❓ Does MA5 > MA10 > MA20 (bullish alignment) hold?
    2. ❓ Is the current MA5 bias within the safe range (< 5%)? — If > 5%, flag as "strictly avoid chasing highs"
    3. ❓ Is volume confirming the move (low-volume pullback / high-volume breakout)?
    4. ❓ Is the chip structure healthy?
    5. ❓ Are there any major negative catalysts? (share reductions, penalties, earnings miss, etc.)

    ### Decision Dashboard Requirements:
    - **Stock Name**: Output the correct full name (e.g., "Kweichow Moutai", not "Stock 600519")
    - **Core Conclusion**: One sentence — buy, sell, or wait
    - **Split Position Advice**: Separate guidance for those with vs. without a position
    - **Precise Sniper Points**: Entry price, stop-loss, and target price (to the cent)
    - **Action Checklist**: Mark each item with ✅ / ⚠️ / ❌

    Output the complete JSON Decision Dashboard. All text values must be in English."""

        return prompt

    
    def _format_volume(self, volume: Optional[float]) -> str:
        """Format trading volume for display."""
        if volume is None:
            return 'N/A'
        if volume >= 1e8:
            return f"{volume / 1e8:.2f} B shares"
        elif volume >= 1e4:
            return f"{volume / 1e4:.2f} W shares"
        else:
            return f"{volume:.0f} shares"
    
    def _format_amount(self, amount: Optional[float]) -> str:
        """Format trading amount for display."""
        if amount is None:
            return 'N/A'
        if amount >= 1e8:
            return f"{amount / 1e8:.2f} B CNY"
        elif amount >= 1e4:
            return f"{amount / 1e4:.2f} W CNY"
        else:
            return f"{amount:.0f} CNY"

    def _format_percent(self, value: Optional[float]) -> str:
        """Format percentage value for display."""
        if value is None:
            return 'N/A'
        try:
            return f"{float(value):.2f}%"
        except (TypeError, ValueError):
            return 'N/A'

    def _format_price(self, value: Optional[float]) -> str:
        """Format price value for display."""
        if value is None:
            return 'N/A'
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return 'N/A'

    def _build_market_snapshot(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build the current-day market snapshot for display."""
        today = context.get('today', {}) or {}
        realtime = context.get('realtime', {}) or {}
        yesterday = context.get('yesterday', {}) or {}

        prev_close = yesterday.get('close')
        close = today.get('close')
        high = today.get('high')
        low = today.get('low')

        amplitude = None
        change_amount = None
        if prev_close not in (None, 0) and high is not None and low is not None:
            try:
                amplitude = (float(high) - float(low)) / float(prev_close) * 100
            except (TypeError, ValueError, ZeroDivisionError):
                amplitude = None
        if prev_close is not None and close is not None:
            try:
                change_amount = float(close) - float(prev_close)
            except (TypeError, ValueError):
                change_amount = None

        snapshot = {
            "date": context.get('date', 'Unknown'),
            "close": self._format_price(close),
            "open": self._format_price(today.get('open')),
            "high": self._format_price(high),
            "low": self._format_price(low),
            "prev_close": self._format_price(prev_close),
            "pct_chg": self._format_percent(today.get('pct_chg')),
            "change_amount": self._format_price(change_amount),
            "amplitude": self._format_percent(amplitude),
            "volume": self._format_volume(today.get('volume')),
            "amount": self._format_amount(today.get('amount')),
        }

        if realtime:
            snapshot.update({
                "price": self._format_price(realtime.get('price')),
                "volume_ratio": realtime.get('volume_ratio', 'N/A'),
                "turnover_rate": self._format_percent(realtime.get('turnover_rate')),
                "source": getattr(realtime.get('source'), 'value', realtime.get('source', 'N/A')),
            })

        return snapshot

    def _check_content_integrity(self, result: AnalysisResult) -> Tuple[bool, List[str]]:
        """Delegate to module-level check_content_integrity."""
        return check_content_integrity(result)

    def _build_integrity_complement_prompt(self, missing_fields: List[str]) -> str:
        """Build complement instruction for missing mandatory fields."""
        lines = ["### Completion requirement: Based on the analysis above, please supplement the following required fields and output the complete JSON:"]
        for f in missing_fields:
            if f == "sentiment_score":
                lines.append("- sentiment_score: overall score 0-100")
            elif f == "operation_advice":
                lines.append("- operation_advice: Buy / Add / Hold / Reduce / Sell / Watch")
            elif f == "analysis_summary":
                lines.append("- analysis_summary: comprehensive analysis summary")
            elif f == "dashboard.core_conclusion.one_sentence":
                lines.append("- dashboard.core_conclusion.one_sentence: one-sentence decision")
            elif f == "dashboard.intelligence.risk_alerts":
                lines.append("- dashboard.intelligence.risk_alerts: list of risk alerts (may be empty array)")
            elif f == "dashboard.battle_plan.sniper_points.stop_loss":
                lines.append("- dashboard.battle_plan.sniper_points.stop_loss: stop-loss price")
        return "\n".join(lines)

    def _build_integrity_retry_prompt(
        self,
        base_prompt: str,
        previous_response: str,
        missing_fields: List[str],
    ) -> str:
        """Build retry prompt using the previous response as the complement baseline."""
        complement = self._build_integrity_complement_prompt(missing_fields)
        previous_output = previous_response.strip()
        return "\n\n".join([
            base_prompt,
            "### Previous output is shown below. Please complete the missing fields based on this output and re-output the complete JSON. Do not omit existing fields:",
            previous_output,
            complement,
        ])

    def _apply_placeholder_fill(self, result: AnalysisResult, missing_fields: List[str]) -> None:
        """Delegate to module-level apply_placeholder_fill."""
        apply_placeholder_fill(result, missing_fields)

    def _parse_response(
        self, 
        response_text: str, 
        code: str, 
        name: str
    ) -> AnalysisResult:
        """
        Parse the Gemini response (decision dashboard version).

        Attempts to extract a JSON-formatted analysis result containing the
        dashboard field. Falls back to intelligent text extraction or a default
        result on failure.
        """
        try:
            # Strip markdown code-fence markers
            cleaned_text = response_text
            if '```json' in cleaned_text:
                cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
            elif '```' in cleaned_text:
                cleaned_text = cleaned_text.replace('```', '')
            
            # Locate JSON content
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = cleaned_text[json_start:json_end]
                
                # Attempt to fix common JSON issues
                json_str = self._fix_json_string(json_str)
                
                data = json.loads(json_str)

                # Schema validation (lenient: on failure, continue with raw dict)
                try:
                    AnalysisReportSchema.model_validate(data)
                except Exception as e:
                    logger.warning(
                        "LLM report schema validation failed, continuing with raw dict: %s",
                        str(e)[:100],
                    )

                # Extract dashboard data
                dashboard = data.get('dashboard', None)

                # Prefer AI-returned stock name when the local name looks invalid
                ai_stock_name = data.get('stock_name')
                if ai_stock_name and (name.startswith('Stock') or name == code or 'Unknown' in name):
                    name = ai_stock_name

                # Parse all fields with safe defaults
                # Derive decision_type from operation_advice when absent
                decision_type = data.get('decision_type', '')
                if not decision_type:
                    op = data.get('operation_advice', 'Hold')
                    if op in ['Buy', 'Add', 'Strong Buy', '买入', '加仓', '强烈买入']:
                        decision_type = 'buy'
                    elif op in ['Sell', 'Reduce', 'Strong Sell', '卖出', '减仓', '强烈卖出']:
                        decision_type = 'sell'
                    else:
                        decision_type = 'hold'
                
                return AnalysisResult(
                    code=code,
                    name=name,
                    # Core metrics
                    sentiment_score=int(data.get('sentiment_score', 50)),
                    trend_prediction=data.get('trend_prediction', 'Neutral'),
                    operation_advice=data.get('operation_advice', 'Hold'),
                    decision_type=decision_type,
                    confidence_level=data.get('confidence_level', 'Medium'),
                    # Decision dashboard
                    dashboard=dashboard,
                    # Trend analysis
                    trend_analysis=data.get('trend_analysis', ''),
                    short_term_outlook=data.get('short_term_outlook', ''),
                    medium_term_outlook=data.get('medium_term_outlook', ''),
                    # Technical
                    technical_analysis=data.get('technical_analysis', ''),
                    ma_analysis=data.get('ma_analysis', ''),
                    volume_analysis=data.get('volume_analysis', ''),
                    pattern_analysis=data.get('pattern_analysis', ''),
                    # Fundamental
                    fundamental_analysis=data.get('fundamental_analysis', ''),
                    sector_position=data.get('sector_position', ''),
                    company_highlights=data.get('company_highlights', ''),
                    # Sentiment / news
                    news_summary=data.get('news_summary', ''),
                    market_sentiment=data.get('market_sentiment', ''),
                    hot_topics=data.get('hot_topics', ''),
                    # Composite
                    analysis_summary=data.get('analysis_summary', 'Analysis complete'),
                    key_points=data.get('key_points', ''),
                    risk_warning=data.get('risk_warning', ''),
                    buy_reason=data.get('buy_reason', ''),
                    # Metadata
                    search_performed=data.get('search_performed', False),
                    data_sources=data.get('data_sources', 'Technical data'),
                    success=True,
                )
            else:
                # No JSON found — fall back to plain-text extraction
                logger.warning("Could not extract JSON from response, falling back to text analysis")
                return self._parse_text_response(response_text, code, name)
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}, falling back to text extraction")
            return self._parse_text_response(response_text, code, name)
    
    def _fix_json_string(self, json_str: str) -> str:
        """Fix common JSON formatting issues."""
        import re
        
        # Remove comments
        json_str = re.sub(r'//.*?\n', '\n', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Ensure boolean literals are lowercase
        json_str = json_str.replace('True', 'true').replace('False', 'false')
        
        # fix by json-repair
        json_str = repair_json(json_str)
        
        return json_str
    
    def _parse_text_response(
        self, 
        response_text: str, 
        code: str, 
        name: str
    ) -> AnalysisResult:
        """Extract analysis information from a plain-text response as best as possible."""
        sentiment_score = 50
        trend = 'Neutral'
        advice = 'Hold'
        
        text_lower = response_text.lower()
        
        # Simple sentiment detection via keywords
        positive_keywords = ['bullish', 'buy', 'uptrend', 'breakout', 'strong', 'positive', 'add', '看多', '买入', '上涨', '突破', '强势', '利好', '加仓']
        negative_keywords = ['bearish', 'sell', 'downtrend', 'breakdown', 'weak', 'negative', 'reduce', '看空', '卖出', '下跌', '跌破', '弱势', '利空', '减仓']
        
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        if positive_count > negative_count + 1:
            sentiment_score = 65
            trend = 'Bullish'
            advice = 'Buy'
            decision_type = 'buy'
        elif negative_count > positive_count + 1:
            sentiment_score = 35
            trend = 'Bearish'
            advice = 'Sell'
            decision_type = 'sell'
        else:
            decision_type = 'hold'
        
        # Use first 500 chars as summary
        summary = response_text[:500] if response_text else 'No analysis result'
        
        return AnalysisResult(
            code=code,
            name=name,
            sentiment_score=sentiment_score,
            trend_prediction=trend,
            operation_advice=advice,
            decision_type=decision_type,
            confidence_level='Low',
            analysis_summary=summary,
            key_points='JSON parsing failed, for reference only',
            risk_warning='Analysis result may be inaccurate; cross-check with other sources',
            raw_response=response_text,
            success=True,
        )
    
    def batch_analyze(
        self, 
        contexts: List[Dict[str, Any]],
        delay_between: float = 2.0
    ) -> List[AnalysisResult]:
        """
        Analyze multiple stocks in batch.

        Note: A delay is inserted between each analysis to avoid API rate limits.

        Args:
            contexts: List of context data dicts
            delay_between: Delay in seconds between each analysis

        Returns:
            List of AnalysisResult
        """
        results = []
        
        for i, context in enumerate(contexts):
            if i > 0:
                logger.debug(f"Waiting {delay_between}s before next analysis...")
                time.sleep(delay_between)
            
            result = self.analyze(context)
            results.append(result)
        
        return results


# Convenience function
def get_analyzer() -> GeminiAnalyzer:
    """Return a GeminiAnalyzer instance."""
    return GeminiAnalyzer()


if __name__ == "__main__":
    # Test / smoke-test block
    logging.basicConfig(level=logging.DEBUG)
    
    # Simulated context data
    test_context = {
        'code': '600519',
        'date': '2026-01-09',
        'today': {
            'open': 1800.0,
            'high': 1850.0,
            'low': 1780.0,
            'close': 1820.0,
            'volume': 10000000,
            'amount': 18200000000,
            'pct_chg': 1.5,
            'ma5': 1810.0,
            'ma10': 1800.0,
            'ma20': 1790.0,
            'volume_ratio': 1.2,
        },
        'ma_status': 'Bullish alignment 📈',
        'volume_change_ratio': 1.3,
        'price_change_ratio': 1.5,
    }
    
    analyzer = GeminiAnalyzer()
    
    if analyzer.is_available():
        print("=== AI Analysis Test ===")
        result = analyzer.analyze(test_context)
        print(f"Analysis result: {result.to_dict()}")
    else:
        print("Gemini API not configured, skipping test")
