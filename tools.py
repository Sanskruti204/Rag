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
    """Fetches detailed real-time stock data with dates, highs, lows, and market info."""
    logger.info(f"[CHART] Calling yfinance_tool for query: {query}")
    try:
        # Try to extract company name/ticker from query
        query_clean = query.replace('stock price of', '').replace('what is the price of', '').replace('price of', '').strip().upper()
        
        # Common ticker mappings for Indian companies and others
        ticker_mappings = {
            'TATA MOTORS': 'TATAMOTORS.NS',
            'TATAMOTORS': 'TATAMOTORS.NS',
            'RELIANCE': 'RELIANCE.NS',
            'INFOSYS': 'INFY',
            'WIPRO': 'WIPRO.NS',
            'HDFC': 'HDFCBANK.NS',
            'HDFC BANK': 'HDFCBANK.NS',
            'ICICI': 'ICICIBANK.NS',
            'ICICI BANK': 'ICICIBANK.NS',
            'SBI': 'SBIN.NS',
            'AXIS BANK': 'AXISBANK.NS',
        }
        
        ticker = None
        
        # Check for direct mappings first
        for company, symbol in ticker_mappings.items():
            if company in query_clean:
                ticker = symbol
                logger.info(f"[SEARCH] Matched '{company}' to ticker '{ticker}'")
                break
        
        # If no mapping found, try to extract first word as ticker
        if not ticker:
            words = query_clean.split()
            blacklist = ["WHAT", "IS", "THE", "PRICE", "STOCK", "OF", "PLEASE", "CURRENT", "FOR", "TELL"]
            for word in words:
                if word not in blacklist and len(word) >= 1:
                    ticker = word
                    logger.info(f"[SEARCH] Extracted ticker from query: {ticker}")
                    break
        
        if not ticker:
            logger.warning("[WARN] Could not extract ticker from query")
            return ""
        
        logger.info(f"[UP] Fetching detailed price data for {ticker}")
        stock = yf.Ticker(ticker)
        
        # Get current price and detailed info
        current_price = stock.fast_info.get('last_price') or stock.info.get('currentPrice')
        
        if current_price is None:
            logger.warning(f"[WARN] No price found for {ticker}")
            return ""
        
        # Get additional data
        prev_close = stock.fast_info.get('previous_close') or stock.info.get('regularMarketPrice')
        fifty_two_week_high = stock.info.get('fiftyTwoWeekHigh', 'N/A')
        fifty_two_week_low = stock.info.get('fiftyTwoWeekLow', 'N/A')
        market_cap = stock.info.get('marketCap', 'N/A')
        pe_ratio = stock.info.get('trailingPE', 'N/A')
        dividend_yield = stock.info.get('dividendYield', 'N/A')
        currency = stock.info.get('currency', 'USD')
        last_update = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Format market cap
        if market_cap != 'N/A':
            if market_cap >= 1e12:
                market_cap_str = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_str = f"${market_cap/1e9:.2f}B"
            else:
                market_cap_str = f"${market_cap/1e6:.2f}M"
        else:
            market_cap_str = "N/A"
        
        # Format PE ratio
        pe_ratio_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"
        
        # Format dividend yield
        dividend_str = f"{dividend_yield*100:.2f}%" if isinstance(dividend_yield, (int, float)) else "N/A"
        
        # Format change
        if prev_close:
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            change_indicator = "📈" if change >= 0 else "📉"
        else:
            change = 0
            change_pct = 0
            change_indicator = "→"
        
        result = f"""
**{ticker} - Real-time Stock Data**

**Current Price:** ${current_price:.2f} {change_indicator} ({change:+.2f}, {change_pct:+.2f}%)
**Previous Close:** ${prev_close:.2f} (if available)
**52-Week High:** ${fifty_two_week_high if fifty_two_week_high != 'N/A' else 'N/A'}
**52-Week Low:** ${fifty_two_week_low if fifty_two_week_low != 'N/A' else 'N/A'}
**Market Cap:** {market_cap_str}
**P/E Ratio:** {pe_ratio_str}
**Dividend Yield:** {dividend_str}
**Last Updated:** {last_update}
"""
        
        logger.info(f"[OK] Got detailed price data for {ticker}")
        return result.strip()
    except Exception as e:
        logger.error(f"[ERROR] Error in yfinance_tool: {str(e)}")
        return ""
    except Exception as e:
        logger.error(f"[ERROR] Error in yfinance_tool: {str(e)}")
        return "NO_TICKER"

def web_search_full(query, llm):
    """Fallback search when local documents lack answers. Returns answer with sources."""
    logger.info(f"[WEB] Calling web_search_full for query: {query}")
    search = DuckDuckGoSearchRun()
    unique_urls = []
    sources_text = ""
    
    try:
        logger.info("[LINK] Performing DuckDuckGo search...")
        search_data = search.run(f"{query}")
        context = search_data if search_data else "No live web results found."
        logger.info(f"[OK] Web search completed. Results length: {len(context)} chars")
        
        # Extract URLs and source information from search results
        import re as regex
        
        # Extract all URLs
        urls = regex.findall(r'https?://[^\s\)]+', search_data) if search_data else []
        unique_urls = list(dict.fromkeys(urls))  # Remove duplicates while preserving order
        logger.info(f"[SOURCES] Found {len(unique_urls)} source URL(s): {unique_urls}")
        
        # Extract source names/titles if available
        source_pattern = r'\[([^\]]+)\]\s*\(https?://[^\)]+\)'
        sources = regex.findall(source_pattern, search_data)
        
        if sources:
            sources_text = "\n\nBased on sources: " + ", ".join(set(sources))
        
    except Exception as e:
        logger.error(f"[ERROR] Web search timeout/error: {str(e)}")
        context = "I encountered a timeout while searching the web. This might be a temporary connectivity issue. Please try again or rephrase your question."
        unique_urls = []
        sources_text = ""

    logger.info("[BRAIN] Generating concise answer from web search context via LLM")
    prompt = f"""Based on the following search results, provide a direct, concise answer to the question.
Be brief and to the point. Only include the most relevant information.
IMPORTANT: Do NOT mention the source names or URLs in your answer. The sources will be displayed separately below.

Search Results:
{context}

Question: {query}

Provide a concise, direct answer (1-2 sentences max) without mentioning sources:"""
    
    result = llm.invoke(prompt).strip()
    logger.info(f"[OK] Web answer generated: {result[:100]}...")
    
    return result, unique_urls, sources_text