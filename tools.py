import datetime
import re
import logging
import yfinance as yf
from langchain_community.tools import DuckDuckGoSearchRun

# Setup logger
logger = logging.getLogger(__name__)

def finance_advisor_tool(query, llm, context_docs=""):
    """Acts as a professional advisor, combining market data and user documents."""
    logger.info("[ADVISOR] Calling finance_advisor_tool")
    today = datetime.datetime.now().strftime('%B %d, %Y')
    logger.info(f"[DATE] Current date: {today}")
    
    prompt = f"""
    You are a Certified Senior Financial Advisor. Provide balanced, risk-aware guidance.
    
    Current Date: {today}
    User Query: {query}
    Local Document Context: {context_docs}
    
    Guidelines:
    1. Tone: Professional and objective.
    2. Risk Disclosure: Always state "Investing involves risk" when discussing markets.
    3. Actionable Advice: Provide clear strategic steps based on context.
    
    Structure: Analysis -> Strategic Advice -> Risk Considerations.
    """
    logger.info("[BRAIN] Generating advisor recommendation via LLM")
    result = llm.invoke(prompt).strip()
    logger.info(f"[OK] Advisor recommendation generated: {result[:100]}...")
    return result

def yfinance_tool(query):
    """Fetches real-time price data if a ticker is found."""
    logger.info(f"[CHART] Calling yfinance_tool for query: {query}")
    try:
        blacklist = ["WHAT", "IS", "THE", "PRICE", "STOCK"]
        words = re.findall(r"\b[A-Z]{2,5}\b", query.upper())
        logger.info(f"[SEARCH] Found potential tickers: {words}")
        
        ticker = next((w for w in words if w not in blacklist), None)
        logger.info(f"[OK] Selected ticker: {ticker}")
        
        if not ticker:
            logger.warning("[WARN] No ticker found")
            return "NO_TICKER"
        
        logger.info(f"[UP] Fetching price for {ticker}")
        stock = yf.Ticker(ticker)
        price = stock.fast_info.get('last_price') or stock.info.get('currentPrice')
        
        if price is None:
            logger.warning(f"[WARN] No price data found for {ticker}")
            return "NO_TICKER"
        
        logger.info(f"[OK] Got price for {ticker}: ${price:.2f}")
        return f"The current price of **{ticker}** is **${price:.2f}**."
    except Exception as e:
        logger.error(f"[ERROR] Error in yfinance_tool: {str(e)}")
        return "NO_TICKER"

def web_search_full(query, llm):
    """Fallback search when local documents lack answers."""
    logger.info(f"[WEB] Calling web_search_full for query: {query}")
    search = DuckDuckGoSearchRun()
    today_str = "February 10, 2026"
    
    try:
        logger.info("[LINK] Performing DuckDuckGo search...")
        search_data = search.run(f"{query} latest update {today_str}")
        context = search_data if search_data else "No live web results found."
        logger.info(f"[OK] Web search completed. Results length: {len(context)} chars")
    except Exception as e:
        logger.error(f"[ERROR] Web search error: {str(e)}")
        context = f"Search error: {str(e)}"

    logger.info("[BRAIN] Generating answer from web search context via LLM")
    prompt = f"Today is {today_str}. Answer based on: {context}\nQuestion: {query}"
    result = llm.invoke(prompt).strip()
    logger.info(f"[OK] Web answer generated: {result[:100]}...")
    return result