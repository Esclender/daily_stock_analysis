# -*- coding: utf-8 -*-
"""
===================================
Market Daily Recap Module
===================================

Responsibilities:
1. Fetch major index data (SSE, SZSE, ChiNext, and US equivalents)
2. Search market news to build recap intelligence
3. Use an LLM to generate the daily market recap report
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd

from src.config import get_config
from src.search_service import SearchService
from src.core.market_profile import get_profile, MarketProfile
from src.core.market_strategy import get_market_strategy_blueprint
from data_provider.base import DataFetcherManager

logger = logging.getLogger(__name__)


@dataclass
class MarketIndex:
    """Market index snapshot."""
    code: str                    # index code
    name: str                    # index name
    current: float = 0.0         # latest level
    change: float = 0.0          # point change
    change_pct: float = 0.0      # percentage change (%)
    open: float = 0.0            # open level
    high: float = 0.0            # high level
    low: float = 0.0             # low level
    prev_close: float = 0.0      # previous close
    volume: float = 0.0          # volume (lots)
    amount: float = 0.0          # turnover (CNY)
    amplitude: float = 0.0       # amplitude (%)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'current': self.current,
            'change': self.change,
            'change_pct': self.change_pct,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'amount': self.amount,
            'amplitude': self.amplitude,
        }


@dataclass
class MarketOverview:
    """Market overview snapshot."""
    date: str                           # date
    indices: List[MarketIndex] = field(default_factory=list)  # major indices
    up_count: int = 0                   # advancing issues
    down_count: int = 0                 # declining issues
    flat_count: int = 0                 # unchanged issues
    limit_up_count: int = 0             # limit-up count
    limit_down_count: int = 0           # limit-down count
    total_amount: float = 0.0           # total market turnover (bn CNY)
    # north_flow: float = 0.0           # northbound net inflow (bn CNY) — deprecated, API unavailable

    # Sector rankings
    top_sectors: List[Dict] = field(default_factory=list)     # top 5 gaining sectors
    bottom_sectors: List[Dict] = field(default_factory=list)  # top 5 losing sectors


class MarketAnalyzer:
    """
    Market daily recap analyzer.

    Capabilities:
    1. Fetch real-time major index quotes
    2. Fetch market advance/decline statistics
    3. Fetch sector gain/loss rankings
    4. Search market news
    5. Generate the daily market recap report
    """

    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        analyzer=None,
        region: str = "cn",
    ):
        """
        Initialize the market analyzer.

        Args:
            search_service: search service instance
            analyzer: AI analyzer instance (used to call the LLM)
            region: market region — cn=A-share, us=US equities
        """
        self.config = get_config()
        self.search_service = search_service
        self.analyzer = analyzer
        self.data_manager = DataFetcherManager()
        self.region = region if region in ("cn", "us") else "cn"
        self.profile: MarketProfile = get_profile(self.region)
        self.strategy = get_market_strategy_blueprint(self.region)

    def get_market_overview(self) -> MarketOverview:
        """
        Fetch the market overview data.

        Returns:
            MarketOverview: market overview data object
        """
        today = datetime.now().strftime('%Y-%m-%d')
        overview = MarketOverview(date=today)

        # 1. Fetch major index quotes (switches between CN and US by region)
        overview.indices = self._get_main_indices()

        # 2. Fetch advance/decline stats (CN only — no US equivalent)
        if self.profile.has_market_stats:
            self._get_market_statistics(overview)

        # 3. Fetch sector rankings (CN only — US not yet supported)
        if self.profile.has_sector_rankings:
            self._get_sector_rankings(overview)

        # 4. Northbound flow (optional, disabled)
        # self._get_north_flow(overview)

        return overview

    
    def _get_main_indices(self) -> List[MarketIndex]:
        """获取主要指数实时行情"""
        indices = []

        try:
            logger.info("[大盘] 获取主要指数实时行情...")

            # 使用 DataFetcherManager 获取指数行情（按 region 切换）
            data_list = self.data_manager.get_main_indices(region=self.region)

            if data_list:
                for item in data_list:
                    index = MarketIndex(
                        code=item['code'],
                        name=item['name'],
                        current=item['current'],
                        change=item['change'],
                        change_pct=item['change_pct'],
                        open=item['open'],
                        high=item['high'],
                        low=item['low'],
                        prev_close=item['prev_close'],
                        volume=item['volume'],
                        amount=item['amount'],
                        amplitude=item['amplitude']
                    )
                    indices.append(index)

            if not indices:
                logger.warning("[大盘] 所有行情数据源失败，将依赖新闻搜索进行分析")
            else:
                logger.info(f"[大盘] 获取到 {len(indices)} 个指数行情")

        except Exception as e:
            logger.error(f"[大盘] 获取指数行情失败: {e}")

        return indices

    def _get_market_statistics(self, overview: MarketOverview):
        """获取市场涨跌统计"""
        try:
            logger.info("[大盘] 获取市场涨跌统计...")

            stats = self.data_manager.get_market_stats()

            if stats:
                overview.up_count = stats.get('up_count', 0)
                overview.down_count = stats.get('down_count', 0)
                overview.flat_count = stats.get('flat_count', 0)
                overview.limit_up_count = stats.get('limit_up_count', 0)
                overview.limit_down_count = stats.get('limit_down_count', 0)
                overview.total_amount = stats.get('total_amount', 0.0)

                logger.info(f"[大盘] 涨:{overview.up_count} 跌:{overview.down_count} 平:{overview.flat_count} "
                          f"涨停:{overview.limit_up_count} 跌停:{overview.limit_down_count} "
                          f"成交额:{overview.total_amount:.0f}亿")

        except Exception as e:
            logger.error(f"[大盘] 获取涨跌统计失败: {e}")

    def _get_sector_rankings(self, overview: MarketOverview):
        """获取板块涨跌榜"""
        try:
            logger.info("[大盘] 获取板块涨跌榜...")

            top_sectors, bottom_sectors = self.data_manager.get_sector_rankings(5)

            if top_sectors or bottom_sectors:
                overview.top_sectors = top_sectors
                overview.bottom_sectors = bottom_sectors

                logger.info(f"[大盘] 领涨板块: {[s['name'] for s in overview.top_sectors]}")
                logger.info(f"[大盘] 领跌板块: {[s['name'] for s in overview.bottom_sectors]}")

        except Exception as e:
            logger.error(f"[大盘] 获取板块涨跌榜失败: {e}")
    
    # def _get_north_flow(self, overview: MarketOverview):
    #     """获取北向资金流入"""
    #     try:
    #         logger.info("[大盘] 获取北向资金...")
    #         
    #         # 获取北向资金数据
    #         df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
    #         
    #         if df is not None and not df.empty:
    #             # 取最新一条数据
    #             latest = df.iloc[-1]
    #             if '当日净流入' in df.columns:
    #                 overview.north_flow = float(latest['当日净流入']) / 1e8  # 转为亿元
    #             elif '净流入' in df.columns:
    #                 overview.north_flow = float(latest['净流入']) / 1e8
    #                 
    #             logger.info(f"[大盘] 北向资金净流入: {overview.north_flow:.2f}亿")
    #             
    #     except Exception as e:
    #         logger.warning(f"[大盘] 获取北向资金失败: {e}")
    
    def search_market_news(self) -> List[Dict]:
        """
        搜索市场新闻
        
        Returns:
            新闻列表
        """
        if not self.search_service:
            logger.warning("[大盘] 搜索服务未配置，跳过新闻搜索")
            return []
        
        all_news = []

        # 按 region 使用不同的新闻搜索词
        search_queries = self.profile.news_queries
        
        try:
            logger.info("[大盘] 开始搜索市场新闻...")
            
            # 根据 region 设置搜索上下文名称，避免美股搜索被解读为 A 股语境
            market_name = "大盘" if self.region == "cn" else "US market"
            for query in search_queries:
                response = self.search_service.search_stock_news(
                    stock_code="market",
                    stock_name=market_name,
                    max_results=3,
                    focus_keywords=query.split()
                )
                if response and response.results:
                    all_news.extend(response.results)
                    logger.info(f"[大盘] 搜索 '{query}' 获取 {len(response.results)} 条结果")
            
            logger.info(f"[大盘] 共获取 {len(all_news)} 条市场新闻")
            
        except Exception as e:
            logger.error(f"[大盘] 搜索市场新闻失败: {e}")
        
        return all_news
    
    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        """
        使用大模型生成大盘复盘报告
        
        Args:
            overview: 市场概览数据
            news: 市场新闻列表 (SearchResult 对象列表)
            
        Returns:
            大盘复盘报告文本
        """
        if not self.analyzer or not self.analyzer.is_available():
            logger.warning("[大盘] AI分析器未配置或不可用，使用模板生成报告")
            return self._generate_template_review(overview, news)
        
        # 构建 Prompt
        prompt = self._build_review_prompt(overview, news)
        
        logger.info("[大盘] 调用大模型生成复盘报告...")
        # Use the public generate_text() entry point — never access private analyzer attributes.
        review = self.analyzer.generate_text(prompt, max_tokens=2048, temperature=0.7)

        if review:
            logger.info("[大盘] 复盘报告生成成功，长度: %d 字符", len(review))
            # Inject structured data tables into LLM prose sections
            return self._inject_data_into_review(review, overview)
        else:
            logger.warning("[大盘] 大模型返回为空，使用模板报告")
            return self._generate_template_review(overview, news)
    
    def _inject_data_into_review(self, review: str, overview: MarketOverview) -> str:
        """Inject structured data tables into the corresponding LLM prose sections."""
        import re

        # Build data blocks
        stats_block = self._build_stats_block(overview)
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview)

        # Inject market stats after "### I. Market Summary" section (before next ###)
        if stats_block:
            review = self._insert_after_section(review, r'###\s*I\.?\s*Market Summary', stats_block)

        # Inject indices table after "### II. Index Commentary" section
        if indices_block:
            review = self._insert_after_section(review, r'###\s*II\.?\s*Index Commentary', indices_block)

        # Inject sector rankings after "### IV. Sector/Theme Highlights" section
        if sector_block:
            review = self._insert_after_section(review, r'###\s*IV\.?\s*Sector', sector_block)

        return review

    @staticmethod
    def _insert_after_section(text: str, heading_pattern: str, block: str) -> str:
        """Insert a data block at the end of a markdown section (before the next ### heading)."""
        import re
        # Find the heading
        match = re.search(heading_pattern, text)
        if not match:
            return text
        start = match.end()
        # Find the next ### heading after this one
        next_heading = re.search(r'\n###\s', text[start:])
        if next_heading:
            insert_pos = start + next_heading.start()
        else:
            # No next heading — append at end
            insert_pos = len(text)
        # Insert the block before the next heading, with spacing
        return text[:insert_pos].rstrip() + '\n\n' + block + '\n\n' + text[insert_pos:].lstrip('\n')

    def _build_stats_block(self, overview: MarketOverview) -> str:
        """Build market statistics block."""
        has_stats = overview.up_count or overview.down_count or overview.total_amount
        if not has_stats:
            return ""
        lines = [
            f"> 📈 Up **{overview.up_count}** / Down **{overview.down_count}** / "
            f"Flat **{overview.flat_count}** | "
            f"Limit-up **{overview.limit_up_count}** / Limit-down **{overview.limit_down_count}** | "
            f"Turnover **{overview.total_amount:.0f}** bn CNY"
        ]
        return "\n".join(lines)

    def _build_indices_block(self, overview: MarketOverview) -> str:
        """Build index quote table (amplitude excluded)."""
        if not overview.indices:
            return ""
        lines = [
            "| Index | Last | Change | Turnover (bn) |",
            "|-------|------|--------|---------------|"]
        for idx in overview.indices:
            arrow = "🔴" if idx.change_pct < 0 else "🟢" if idx.change_pct > 0 else "⚪"
            amount_raw = idx.amount or 0.0
            if amount_raw == 0.0:
                # Yahoo Finance does not provide turnover — show N/A to avoid confusion
                amount_str = "N/A"
            elif amount_raw > 1e6:
                amount_str = f"{amount_raw / 1e8:.0f}"
            else:
                amount_str = f"{amount_raw:.0f}"
            lines.append(f"| {idx.name} | {idx.current:.2f} | {arrow} {idx.change_pct:+.2f}% | {amount_str} |")
        return "\n".join(lines)

    def _build_sector_block(self, overview: MarketOverview) -> str:
        """Build sector ranking block."""
        if not overview.top_sectors and not overview.bottom_sectors:
            return ""
        lines = []
        if overview.top_sectors:
            top = " | ".join(
                [f"**{s['name']}**({s['change_pct']:+.2f}%)" for s in overview.top_sectors[:5]]
            )
            lines.append(f"> 🔥 Leading: {top}")
        if overview.bottom_sectors:
            bot = " | ".join(
                [f"**{s['name']}**({s['change_pct']:+.2f}%)" for s in overview.bottom_sectors[:5]]
            )
            lines.append(f"> 💧 Lagging: {bot}")
        return "\n".join(lines)

    def _build_review_prompt(self, overview: MarketOverview, news: List) -> str:
        """Build the market recap prompt."""
        # Index data (compact, no emoji)
        indices_text = ""
        for idx in overview.indices:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- {idx.name}: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"

        # Sector data
        top_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.top_sectors[:3]])
        bottom_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.bottom_sectors[:3]])

        # News — supports both SearchResult objects and dicts
        news_text = ""
        for i, n in enumerate(news[:6], 1):
            if hasattr(n, 'title'):
                title = n.title[:50] if n.title else ''
                snippet = n.snippet[:100] if n.snippet else ''
            else:
                title = n.get('title', '')[:50]
                snippet = n.get('snippet', '')[:100]
            news_text += f"{i}. {title}\n   {snippet}\n"

        # Build stats and sector blocks (US has no advance/decline or sector data)
        stats_block = ""
        sector_block = ""
        if self.profile.has_market_stats:
            stats_block = (
                "## Market Overview\n"
                f"- Up: {overview.up_count} | Down: {overview.down_count} | Flat: {overview.flat_count}\n"
                f"- Limit-up: {overview.limit_up_count} | Limit-down: {overview.limit_down_count}\n"
                f"- Total turnover: {overview.total_amount:.0f} bn CNY"
            )
        else:
            stats_block = "## Market Overview\n(Advance/decline stats not available.)"

        if self.profile.has_sector_rankings:
            sector_block = (
                "## Sector Performance\n"
                f"Leading: {top_sectors_text if top_sectors_text else 'N/A'}\n"
                f"Lagging: {bottom_sectors_text if bottom_sectors_text else 'N/A'}"
            )
        else:
            sector_block = "## Sector Performance\n(Sector data not available.)"

        data_no_indices_hint = (
            "Note: Market data fetch failed. Base your analysis mainly on [Market News] "
            "and avoid inventing specific index levels."
            if not indices_text
            else ""
        )
        indices_placeholder = indices_text if indices_text else "No index data (API error)"
        news_placeholder = news_text if news_text else "No relevant news"

        # US market uses a dedicated US-context prompt
        if self.region == "us":
            return (
                "You are a professional US/A/H market analyst. "
                "Please produce a concise US market recap report based on the data below.\n\n"
                "[Requirements]\n"
                "- Please provide the entire market review in English.\n"
                "- Output pure Markdown only\n"
                "- No JSON\n"
                "- No code blocks\n"
                "- Use emoji sparingly in headings (at most one per heading)\n\n"
                "---\n\n"
                "# Today's Market Data\n\n"
                f"## Date\n{overview.date}\n\n"
                f"## Major Indices\n{indices_placeholder}\n\n"
                f"{stats_block}\n\n"
                f"{sector_block}\n\n"
                f"## Market News\n{news_placeholder}\n\n"
                f"{data_no_indices_hint}\n\n"
                f"{self.strategy.to_prompt_block()}\n\n"
                "---\n\n"
                "# Output Template (follow this structure)\n\n"
                f"## {overview.date} US Market Recap\n\n"
                "### I. Market Summary\n"
                "(2-3 sentences on overall market performance, index moves, volume)\n\n"
                "### II. Index Commentary\n"
                f"({self.profile.prompt_index_hint})\n\n"
                "### III. Fund Flows\n"
                "(Interpret volume and flow implications)\n\n"
                "### IV. Sector/Theme Highlights\n"
                "(Analyze drivers behind leading/lagging sectors)\n\n"
                "### V. Outlook\n"
                "(Short-term view based on price action and news)\n\n"
                "### VI. Risk Alerts\n"
                "(Key risks to watch)\n\n"
                "### VII. Strategy Plan\n"
                "(Provide risk-on/neutral/risk-off stance, position sizing guideline, "
                "and one invalidation trigger.)\n\n"
                "---\n\n"
                "Output the report content directly, no extra commentary.\n"
            )

        # CN market prompt — English output required
        return (
            "You are a professional A/H/US market analyst. "
            "Please produce a concise market recap report based on the data below.\n\n"
            "[Requirements]\n"
            "- Please provide the entire market review in English.\n"
            "- Output pure Markdown only\n"
            "- No JSON\n"
            "- No code blocks\n"
            "- Use emoji sparingly in headings (at most one per heading)\n\n"
            "---\n\n"
            "# Today's Market Data\n\n"
            f"## Date\n{overview.date}\n\n"
            f"## Major Indices\n{indices_placeholder}\n\n"
            f"{stats_block}\n\n"
            f"{sector_block}\n\n"
            f"## Market News\n{news_placeholder}\n\n"
            f"{data_no_indices_hint}\n\n"
            f"{self.strategy.to_prompt_block()}\n\n"
            "---\n\n"
            "# Output Template (follow this structure strictly)\n\n"
            f"## {overview.date} Market Recap\n\n"
            "### I. Market Summary\n"
            "(2-3 sentences on overall market performance: index moves, volume, breadth)\n\n"
            "### II. Index Commentary\n"
            f"({self.profile.prompt_index_hint})\n\n"
            "### III. Fund Flows\n"
            "(Interpret turnover and capital flow implications)\n\n"
            "### IV. Sector/Theme Highlights\n"
            "(Analyze the drivers behind leading and lagging sectors)\n\n"
            "### V. Outlook\n"
            "(Short-term view based on current price action and news)\n\n"
            "### VI. Risk Alerts\n"
            "(Key risks to watch)\n\n"
            "### VII. Strategy Plan\n"
            "(State an offensive/balanced/defensive stance, position sizing guideline, "
            "and one invalidation trigger. "
            "Close with: \"For reference only — not investment advice.\")\n\n"
            "---\n\n"
            "Output the report content directly, no extra commentary.\n"
        )

    
    def _generate_template_review(self, overview: MarketOverview, news: List) -> str:
        """Generate a template-based recap report (fallback when no LLM is available)."""
        mood_code = self.profile.mood_index_code
        # Locate the mood index by code.
        # cn: mood_code="000001", idx.code may be "sh000001" (ends with mood_code)
        # us: mood_code="SPX",    idx.code is "SPX" directly
        mood_index = next(
            (
                idx
                for idx in overview.indices
                if idx.code == mood_code or idx.code.endswith(mood_code)
            ),
            None,
        )
        if mood_index:
            if mood_index.change_pct > 1:
                market_mood = "strong gains"
            elif mood_index.change_pct > 0:
                market_mood = "modest gains"
            elif mood_index.change_pct > -1:
                market_mood = "modest losses"
            else:
                market_mood = "significant losses"
        else:
            market_mood = "mixed / sideways"
        
        # Index quotes (compact format)
        indices_text = ""
        for idx in overview.indices[:4]:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- **{idx.name}**: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"

        # Sector names
        top_text = " / ".join([s['name'] for s in overview.top_sectors[:3]])
        bottom_text = " / ".join([s['name'] for s in overview.bottom_sectors[:3]])
        
        # Include advance/decline stats and sector rankings only when available (CN market)
        stats_section = ""
        if self.profile.has_market_stats:
            stats_section = (
                "\n### III. Advance / Decline Statistics\n"
                "| Metric | Value |\n"
                "|--------|-------|\n"
                f"| Advancing | {overview.up_count} |\n"
                f"| Declining | {overview.down_count} |\n"
                f"| Limit-up | {overview.limit_up_count} |\n"
                f"| Limit-down | {overview.limit_down_count} |\n"
                f"| Total turnover | {overview.total_amount:.0f} bn CNY |\n"
            )
        sector_section = ""
        if self.profile.has_sector_rankings and (top_text or bottom_text):
            sector_section = (
                "\n### IV. Sector Performance\n"
                f"- **Leading**: {top_text if top_text else 'N/A'}\n"
                f"- **Lagging**: {bottom_text if bottom_text else 'N/A'}\n"
            )
        market_label = "CN" if self.region == "cn" else "US"
        strategy_summary = self.strategy.to_markdown_block()
        report = (
            f"## {overview.date} Market Recap\n\n"
            f"### I. Market Summary\n"
            f"Today's {market_label} market closed with **{market_mood}**.\n\n"
            f"### II. Major Indices\n"
            f"{indices_text}\n"
            f"{stats_section}\n"
            f"{sector_section}\n"
            f"### V. Risk Alerts\n"
            f"Markets involve risk. The data above is for reference only and does not constitute investment advice.\n\n"
            f"{strategy_summary}\n\n"
            f"---\n"
            f"*Generated at: {datetime.now().strftime('%H:%M')}*\n"
        )
        return report
    
    def run_daily_review(self) -> str:
        """
        执行每日大盘复盘流程
        
        Returns:
            复盘报告文本
        """
        logger.info("========== 开始大盘复盘分析 ==========")
        
        # 1. 获取市场概览
        overview = self.get_market_overview()
        
        # 2. 搜索市场新闻
        news = self.search_market_news()
        
        # 3. 生成复盘报告
        report = self.generate_market_review(overview, news)
        
        logger.info("========== 大盘复盘分析完成 ==========")
        
        return report


# Test entry point
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )

    analyzer = MarketAnalyzer()

    # Test: fetch market overview
    overview = analyzer.get_market_overview()
    print(f"\n=== Market Overview ===")
    print(f"Date: {overview.date}")
    print(f"Indices: {len(overview.indices)}")
    for idx in overview.indices:
        print(f"  {idx.name}: {idx.current:.2f} ({idx.change_pct:+.2f}%)")
    print(f"Up: {overview.up_count} | Down: {overview.down_count}")
    print(f"Turnover: {overview.total_amount:.0f} bn CNY")

    # Test: generate template report
    report = analyzer._generate_template_review(overview, [])
    print(f"\n=== Market Recap ===")
    print(report)
