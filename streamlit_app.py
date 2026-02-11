import os
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import json
import re
import requests
from bs4 import BeautifulSoup
import time
import logging
from urllib.parse import quote_plus
import datetime
import html
from scrapingbee import ScrapingBeeClient
import requests
from serpapi.google_search import GoogleSearch
from chatbot_rag import get_reddit_rag
from langchain_core.documents import Document
import yfinance as yf

rag = get_reddit_rag()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("company_deepdive")

# Set page configuration
st.set_page_config(
    page_title="Company Deep Dive Chatbot",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "company_data" not in st.session_state:
    st.session_state.company_data = {}

# Security functions
def sanitize_input(text):
    """Sanitize user input to prevent injection attacks"""
    if not text:
        return ""
    # Remove any potential HTML/script tags
    text = re.sub(r'<[^>]*>', '', text)
    # Limit length to prevent DoS
    return text[:1000]

def validate_company_name(company_name):
    """Validate company name to ensure it's a reasonable query"""
    if not company_name:
        return False
    # Check if company name is too short or contains invalid characters
    if len(company_name) < 2 or not re.match(r'^[a-zA-Z0-9\s\.\,\&\-]+$', company_name):
        return False
    return True

# SEC EDGAR API Functions
def search_company(company_name):
    """Search for a company in SEC EDGAR database by name"""
    sanitized_company = sanitize_input(company_name)
    if not validate_company_name(sanitized_company):
        return {"error": "Invalid company name provided"}
    
    try:
        # SEC EDGAR company search API
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?company={quote_plus(sanitized_company)}&owner=exclude&action=getcompany&output=atom"
        headers = {
            "User-Agent": "CompanyDeepDive research@example.com"  # SEC requires a user-agent
        }
        
        logger.info(f"Searching for company: {sanitized_company}")
        logger.info(f"Request URL: {url}")
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Log response status and content length
        logger.info(f"Response status: {response.status_code}, Content length: {len(response.content)}")
        
        # Parse XML response
        soup = BeautifulSoup(response.content, 'lxml-xml')
        
        # Check if company was found
        if "No matching companies" in response.text:
            logger.warning(f"No matching companies found for: {sanitized_company}")
            return {"error": "No matching companies found"}
        
        # Extract company information
        company_info = {}
        
        # Get CIK (Central Index Key) - Try multiple methods
        cik_found = False
        
        # Method 1: Look for CIK in id tag
        cik_tag = soup.find('id')
        if cik_tag:
            cik_match = re.search(r'CIK=(\d+)', cik_tag.text)
            if cik_match:
                company_info['cik'] = cik_match.group(1).zfill(10)  # SEC now uses 10-digit CIKs
                cik_found = True
                logger.info(f"CIK found in id tag: {company_info['cik']}")
        
        # Method 2: Look for CIK in directory tag
        if not cik_found:
            directory_tag = soup.find('directory')
            if directory_tag:
                cik_match = re.search(r'(\d+)', directory_tag.text)
                if cik_match:
                    company_info['cik'] = cik_match.group(1).zfill(10)
                    cik_found = True
                    logger.info(f"CIK found in directory tag: {company_info['cik']}")
        
        # Method 3: Look for CIK in any tag with CIK pattern
        if not cik_found:
            for tag in soup.find_all():
                if tag.string:
                    cik_match = re.search(r'CIK[=:]?\s*(\d+)', tag.string)
                    if cik_match:
                        company_info['cik'] = cik_match.group(1).zfill(10)
                        cik_found = True
                        logger.info(f"CIK found in tag {tag.name}: {company_info['cik']}")
                        break
        
        # Method 4: Look for CIK in href attributes
        if not cik_found:
            for tag in soup.find_all(href=True):
                href = tag.get('href', '')
                cik_match = re.search(r'CIK=(\d+)', href)
                if cik_match:
                    company_info['cik'] = cik_match.group(1).zfill(10)
                    cik_found = True
                    logger.info(f"CIK found in href attribute: {company_info['cik']}")
                    break
        
        # Method 5: Look for CIK in any attribute of any tag
        if not cik_found:
            for tag in soup.find_all():
                for attr_name, attr_value in tag.attrs.items():
                    if isinstance(attr_value, str):
                        cik_match = re.search(r'CIK[=:]?\s*(\d+)', attr_value)
                        if cik_match:
                            company_info['cik'] = cik_match.group(1).zfill(10)
                            cik_found = True
                            logger.info(f"CIK found in {attr_name} attribute: {company_info['cik']}")
                            break
                if cik_found:
                    break
        
        # Method 6: Try to extract from the full response text
        if not cik_found:
            # Look for CIK pattern in the entire response
            cik_matches = re.findall(r'CIK[=:]?\s*(\d+)', response.text)
            if cik_matches:
                company_info['cik'] = cik_matches[0].zfill(10)
                cik_found = True
                logger.info(f"CIK found in full response text: {company_info['cik']}")
            else:
                # Try a more general pattern to find any number that might be a CIK
                # Look for numbers in specific contexts that might indicate a CIK
                cik_matches = re.findall(r'(?:cik|CIK|Central Index Key|company-info)[^0-9]*(\d{5,10})', response.text)
                if cik_matches:
                    company_info['cik'] = cik_matches[0].zfill(10)
                    cik_found = True
                    logger.info(f"CIK found with general pattern: {company_info['cik']}")
        
        # If CIK still not found, try direct API call to the company search JSON endpoint
        if not cik_found:
            logger.warning(f"CIK not found in XML response for: {sanitized_company}")
            logger.debug(f"Response content: {response.text[:1000]}...")  # Log first 1000 chars
            
            # Try the alternative JSON API endpoint
            try:
                alt_url = f"https://www.sec.gov/cgi-bin/browse-edgar?company={quote_plus(sanitized_company)}&owner=exclude&action=getcompany&output=json"
                alt_response = requests.get(alt_url, headers=headers)
                alt_response.raise_for_status()
                
                # Parse JSON response
                json_data = alt_response.json()
                if 'cik' in json_data:
                    company_info['cik'] = str(json_data['cik']).zfill(10)
                    cik_found = True
                    logger.info(f"CIK found in JSON response: {company_info['cik']}")
                elif 'ciks' in json_data and json_data['ciks']:
                    # Get the first CIK if multiple are returned
                    company_info['cik'] = str(list(json_data['ciks'].keys())[0]).zfill(10)
                    cik_found = True
                    logger.info(f"CIK found in JSON ciks field: {company_info['cik']}")
            except Exception as e:
                logger.warning(f"Failed to get CIK from alternative JSON endpoint: {str(e)}")
        
        # Get company name
        name_tag = soup.find('company-info', {'reg-s-k-form'})
        if name_tag:
            company_info['name'] = name_tag.text.strip()
            logger.info(f"Company name found: {company_info['name']}")
        else:
            title_tag = soup.find('title')
            if title_tag:
                company_info['name'] = title_tag.text.strip()
                logger.info(f"Company name found in title: {company_info['name']}")
        
        # Get SIC (Standard Industrial Classification)
        sic_tag = soup.find('assigned-sic')
        if sic_tag:
            company_info['sic'] = sic_tag.text.strip()
            
            # Get SIC description
            sic_desc_tag = soup.find('assigned-sic-desc')
            if sic_desc_tag:
                company_info['sic_description'] = sic_desc_tag.text.strip()
        
        # Get fiscal year end
        fiscal_year_tag = soup.find('fiscal-year-end')
        if fiscal_year_tag:
            company_info['fiscal_year_end'] = fiscal_year_tag.text.strip()
        
        # Get state of incorporation
        state_tag = soup.find('state-of-incorporation')
        if state_tag:
            company_info['state'] = state_tag.text.strip()
        
        # Final check for CIK
        if 'cik' not in company_info:
            logger.error(f"Failed to extract CIK for company: {sanitized_company}")
            return {"error": "Could not extract CIK from SEC response. Please try a different company name or ticker."}
            
        return company_info
    
    except Exception as e:
        logger.error(f"Error searching company: {str(e)}")
        return {"error": f"Failed to search company: {str(e)}"}

def get_company_filings(cik, filing_type=None, limit=10):
    """Get recent filings for a company by CIK"""
    try:
        # SEC EDGAR filings API
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        headers = {
            "User-Agent": "CompanyDeepDive research@example.com"  # SEC requires a user-agent
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract recent filings
        filings = []
        
        if 'filings' in data and 'recent' in data['filings']:
            recent = data['filings']['recent']
            
            # Get the indices of filings
            for i in range(min(limit, len(recent.get('accessionNumber', [])))):
                filing = {
                    'accessionNumber': recent.get('accessionNumber', [])[i],
                    'filingDate': recent.get('filingDate', [])[i],
                    'form': recent.get('form', [])[i],
                    'primaryDocument': recent.get('primaryDocument', [])[i],
                    'reportDate': recent.get('reportDate', [])[i] if i < len(recent.get('reportDate', [])) else None,
                }
                
                # Filter by filing type if specified
                if filing_type is None or filing['form'] == filing_type:
                    filings.append(filing)
                    
                    # Stop when we reach the limit
                    if len(filings) >= limit:
                        break
        
        return filings
    
    except Exception as e:
        logger.error(f"Error getting company filings: {str(e)}")
        return {"error": f"Failed to get company filings: {str(e)}"}

def get_filing_content(cik, accession_number, primary_document):
    """Get the content of a specific filing"""
    try:
        # Format accession number for URL (remove dashes)
        accession_number_clean = accession_number.replace('-', '')
        
        # SEC EDGAR filing content URL
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number_clean}/{primary_document}"
        headers = {
            "User-Agent": "CompanyDeepDive research@example.com"  # SEC requires a user-agent
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Return the text content
        return soup.get_text()
    
    except Exception as e:
        logger.error(f"Error getting filing content: {str(e)}")
        return f"Failed to get filing content: {str(e)}"

def extract_financial_data(cik):
    """Extract key financial data from company filings"""
    try:
        # Get company facts from SEC API
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        headers = {
            "User-Agent": "CompanyDeepDive research@example.com"  # SEC requires a user-agent
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract key financial metrics
        financial_data = {
            "revenue": [],
            "netIncome": [],
            "totalAssets": [],
            "totalLiabilities": []
        }
        
        if 'facts' in data and 'us-gaap' in data['facts']:
            us_gaap = data['facts']['us-gaap']
            
            # Revenue (try different possible tags)
            for revenue_tag in ['Revenue', 'Revenues', 'SalesRevenueNet', 'SalesRevenueGoodsNet']:
                if revenue_tag in us_gaap:
                    for unit in us_gaap[revenue_tag]['units']:
                        if unit == 'USD':
                            for value in us_gaap[revenue_tag]['units'][unit]:
                                if 'form' in value and value['form'] == '10-K':
                                    financial_data['revenue'].append({
                                        'date': value['end'],
                                        'value': value['val']
                                    })
            
            # Net Income
            if 'NetIncomeLoss' in us_gaap:
                for unit in us_gaap['NetIncomeLoss']['units']:
                    if unit == 'USD':
                        for value in us_gaap['NetIncomeLoss']['units'][unit]:
                            if 'form' in value and value['form'] == '10-K':
                                financial_data['netIncome'].append({
                                    'date': value['end'],
                                    'value': value['val']
                                })
            
            # Total Assets
            if 'Assets' in us_gaap:
                for unit in us_gaap['Assets']['units']:
                    if unit == 'USD':
                        for value in us_gaap['Assets']['units'][unit]:
                            if 'form' in value and value['form'] == '10-K':
                                financial_data['totalAssets'].append({
                                    'date': value['end'],
                                    'value': value['val']
                                })
            
            # Total Liabilities
            if 'Liabilities' in us_gaap:
                for unit in us_gaap['Liabilities']['units']:
                    if unit == 'USD':
                        for value in us_gaap['Liabilities']['units'][unit]:
                            if 'form' in value and value['form'] == '10-K':
                                financial_data['totalLiabilities'].append({
                                    'date': value['end'],
                                    'value': value['val']
                                })
        
        # Sort data by date
        for key in financial_data:
            financial_data[key] = sorted(financial_data[key], key=lambda x: x['date'], reverse=True)
        
        return financial_data
    
    except Exception as e:
        logger.error(f"Error extracting financial data: {str(e)}")
        return {"error": f"Failed to extract financial data: {str(e)}"}

def extract_company_info(cik):
    """Extract company information from filings"""
    try:
        # Get the most recent 10-K filing
        filings = get_company_filings(cik, filing_type="10-K", limit=1)
        
        if isinstance(filings, dict) and "error" in filings:
            return {"error": filings["error"]}
        
        if not filings:
            return {"error": "No 10-K filings found"}
        
        # Get the content of the 10-K filing
        filing_content = get_filing_content(cik, filings[0]['accessionNumber'], filings[0]['primaryDocument'])
        
        # Extract key sections from the 10-K
        business_section = extract_section(filing_content, "Item 1", "Item 1A")
        risk_factors = extract_section(filing_content, "Item 1A", "Item 1B")
        
        # Get company facts
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        headers = {
            "User-Agent": "CompanyDeepDive research@example.com"
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract company information
        company_info = {
            "name": data.get("name", ""),
            "sic": data.get("sicCode", ""),
            "sicDescription": data.get("sicDescription", ""),
            "website": data.get("website", ""),
            "description": business_section[:5000] if len(business_section) > 5000 else business_section,
            "riskFactors": risk_factors[:5000] if len(risk_factors) > 5000 else risk_factors,
            "filingDate": filings[0]['filingDate'],
            "fiscalYearEnd": data.get("fiscalYearEnd", "")
        }
        
        return company_info
    
    except Exception as e:
        logger.error(f"Error extracting company info: {str(e)}")
        return {"error": f"Failed to extract company info: {str(e)}"}

def extract_section(text, start_marker, end_marker):
    """Extract a section from the filing text"""
    try:
        # Find the start of the section
        start_index = text.find(start_marker)
        if start_index == -1:
            # Try alternative format
            start_index = text.find(start_marker.upper())
            if start_index == -1:
                return ""
        
        # Find the end of the section
        end_index = text.find(end_marker, start_index)
        if end_index == -1:
            # Try alternative format
            end_index = text.find(end_marker.upper(), start_index)
            if end_index == -1:
                # If end marker not found, take a reasonable chunk
                end_index = start_index + 50000
        
        # Extract the section
        section = text[start_index:end_index].strip()
        
        # Clean up the section (remove HTML tags, excessive whitespace, etc.)
        section = re.sub(r'<[^>]*>', ' ', section)
        section = re.sub(r'\s+', ' ', section)
        
        return section
    
    except Exception as e:
        logger.error(f"Error extracting section: {str(e)}")
        return ""

def fetch_company_info(company_name):
    """Fetch company information from SEC EDGAR"""
    sanitized_company = sanitize_input(company_name)
    if not validate_company_name(sanitized_company):
        return {"error": "Invalid company name provided"}
    
    try:
        # Search for the company
        company_search = search_company(sanitized_company)
        
        if "error" in company_search:
            return company_search
        
        if "cik" not in company_search:
            return {"error": "Company CIK not found"}
        
        # Get company CIK
        cik = company_search["cik"]
        
        # Extract company information
        company_info = extract_company_info(cik)
        
        if "error" in company_info:
            return company_info
        
        # Extract financial data
        financial_data = extract_financial_data(cik)
        
        # Combine all data
        result = {
            "cik": cik,
            "name": company_search.get("name", sanitized_company),
            "info": company_info,
            "financials": financial_data
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error fetching company info: {str(e)}")
        return {"error": f"Failed to fetch company information: {str(e)}"}

def analyze_company_sentiment(company_name):
    """Analyze market sentiment about the company using recent filings"""
    sanitized_company = sanitize_input(company_name)
    if not validate_company_name(sanitized_company):
        return {"error": "Invalid company name provided"}
    
    try:
        # Search for the company
        company_search = search_company(sanitized_company)
        
        if "error" in company_search:
            return company_search
        
        if "cik" not in company_search:
            return {"error": "Company CIK not found"}
        
        # Get company CIK
        cik = company_search["cik"]
        
        # Get recent filings (10-K, 10-Q, 8-K)
        filings = get_company_filings(cik, limit=5)
        
        if isinstance(filings, dict) and "error" in filings:
            return {"error": filings["error"]}
        
        # Analyze sentiment from filings
        sentiment_analysis = "Based on recent SEC filings:\n\n"
        
        for filing in filings:
            filing_date = filing['filingDate']
            form_type = filing['form']
            
            sentiment_analysis += f"- {form_type} filed on {filing_date}\n"
            
            # For 8-K filings, try to extract the reason
            if form_type == "8-K":
                filing_content = get_filing_content(cik, filing['accessionNumber'], filing['primaryDocument'])
                
                # Extract the item from the 8-K
                item_match = re.search(r'Item\s+([0-9\.]+)', filing_content)
                if item_match:
                    item = item_match.group(1)
                    sentiment_analysis += f"  - Reported under Item {item}"
                    
                    # Map common 8-K items to descriptions
                    item_descriptions = {
                        "1.01": "Entry into a Material Definitive Agreement",
                        "1.02": "Termination of a Material Definitive Agreement",
                        "2.01": "Completion of Acquisition or Disposition of Assets",
                        "2.02": "Results of Operations and Financial Condition",
                        "2.03": "Creation of a Direct Financial Obligation",
                        "2.04": "Triggering Events That Accelerate or Increase a Direct Financial Obligation",
                        "2.05": "Costs Associated with Exit or Disposal Activities",
                        "2.06": "Material Impairments",
                        "3.01": "Notice of Delisting or Failure to Satisfy a Continued Listing Rule",
                        "3.02": "Unregistered Sales of Equity Securities",
                        "3.03": "Material Modifications to Rights of Security Holders",
                        "4.01": "Changes in Registrant's Certifying Accountant",
                        "4.02": "Non-Reliance on Previously Issued Financial Statements",
                        "5.01": "Changes in Control of Registrant",
                        "5.02": "Departure of Directors or Certain Officers",
                        "5.03": "Amendments to Articles of Incorporation or Bylaws",
                        "5.04": "Temporary Suspension of Trading Under Registrant's Employee Benefit Plans",
                        "5.05": "Amendments to the Registrant's Code of Ethics",
                        "5.06": "Change in Shell Company Status",
                        "5.07": "Submission of Matters to a Vote of Security Holders",
                        "5.08": "Shareholder Director Nominations",
                        "7.01": "Regulation FD Disclosure",
                        "8.01": "Other Events",
                        "9.01": "Financial Statements and Exhibits"
                    }
                    
                    if item in item_descriptions:
                        sentiment_analysis += f" ({item_descriptions[item]})\n"
                    else:
                        sentiment_analysis += "\n"
                else:
                    sentiment_analysis += "\n"
        
        # Add financial trend analysis if available
        financial_data = extract_financial_data(cik)
        
        if not isinstance(financial_data, dict) or "error" in financial_data:
            sentiment_analysis += "\nFinancial data not available for trend analysis."
        else:
            sentiment_analysis += "\nFinancial Trends:\n"
            
            # Revenue trend
            if financial_data["revenue"] and len(financial_data["revenue"]) >= 2:
                latest = financial_data["revenue"][0]
                previous = financial_data["revenue"][1]
                
                change = ((latest["value"] - previous["value"]) / previous["value"]) * 100
                
                sentiment_analysis += f"- Revenue: {'increased' if change > 0 else 'decreased'} by {abs(change):.2f}% "
                sentiment_analysis += f"from {previous['date']} to {latest['date']}\n"
            
            # Net Income trend
            if financial_data["netIncome"] and len(financial_data["netIncome"]) >= 2:
                latest = financial_data["netIncome"][0]
                previous = financial_data["netIncome"][1]
                
                if previous["value"] != 0:
                    change = ((latest["value"] - previous["value"]) / abs(previous["value"])) * 100
                    
                    sentiment_analysis += f"- Net Income: {'increased' if change > 0 else 'decreased'} by {abs(change):.2f}% "
                    sentiment_analysis += f"from {previous['date']} to {latest['date']}\n"
        
        return {"sentiment": sentiment_analysis}
    
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return {"error": f"Failed to analyze company sentiment: {str(e)}"}

def get_earnings_transcript(company_name, year=None, quarter=None):
    """Fetch and analyze earnings call transcript using DefeatBeta"""
    sanitized_company = sanitize_input(company_name)
    if not validate_company_name(sanitized_company):
        return {"error": "Invalid company name provided"}
    
    try:
        # Validate and sanitize year and quarter inputs
        current_year = datetime.datetime.now().year
        
        # If year is not provided, use the current year
        if year is None:
            year = current_year
        else:
            # Ensure year is an integer and within a reasonable range
            try:
                year = int(year)
                if year < 2000 or year > current_year:
                    return {"error": f"Year must be between 2000 and {current_year}"}
            except ValueError:
                return {"error": "Year must be a valid number"}
        
        # If quarter is not provided, use the most recent quarter
        if quarter is None:
            current_month = datetime.datetime.now().month
            quarter = ((current_month - 1) // 3) + 1
        else:
            # Ensure quarter is an integer between 1 and 4
            try:
                quarter = int(quarter)
                if quarter < 1 or quarter > 4:
                    return {"error": "Quarter must be between 1 and 4"}
            except ValueError:
                return {"error": "Quarter must be a valid number"}
    except Exception as e: 
        logger.error(f"Error fetching or analyzing transcript: {str(e)}")
    logger.info(f"Fetching earnings transcript for {sanitized_company} (Year: {year}, Quarter: {quarter}) using DefeatBeta")
    url = "https://www.sec.gov/files/company_tickers.json" 
    response = requests.get(url, headers={'User-Agent': 'your-email@example.com'}) 
    companies = pd.DataFrame.from_dict(response.json(), orient='index') # Search for company 
    result = companies[companies['title'].str.contains(company_name, case=False)] 
    ticker=result['ticker'].values[0]
    query = f"site:fool.com {ticker} Q{quarter} {year} earnings call"
    params = { "engine": "google", "q": query, "api_key": "1b6c33844c034b01987d113928c20e7dc77c934345ae673545479a7b77f8e7c1", "num": 1, } 
    search = GoogleSearch(params) 
    results = search.get_dict() 
    filtered_links = [result["link"] for result in results.get("organic_results", [])]
    url = filtered_links[0]
    if str(year) or str(ticker) or f"q{quarter}" not in url:
        return "Earnings call not available"
    headers = { "User-Agent": "Mozilla/5.0" } 
    response = requests.get(url, headers=headers, timeout=30) 
    html = response.text
    soup = BeautifulSoup(html, "lxml") 
    for tag in soup(["script", "style", "noscript"]): 
        tag.decompose() 
    text = " ".join(soup.get_text().split())
    return url + "\n\n" + "Earnings Retriever Tool:\n\n"+text.split("Full Conference Call Transcript", 1)[1].strip()[:2000]
            


def get_company_swot(company_name):
    """Generate a SWOT analysis for the company based on SEC filings"""
    sanitized_company = sanitize_input(company_name)
    if not validate_company_name(sanitized_company):
        logger.warning(f"Invalid company name provided: {company_name}")
        return {"error": "Invalid company name provided"}
    
    try:
        logger.info(f"Generating SWOT analysis for: {sanitized_company}")
        
        # Search for the company
        company_search = search_company(sanitized_company)
        
        if "error" in company_search:
            logger.warning(f"Error searching for company: {company_search['error']}")
            return company_search
        
        if "cik" not in company_search:
            logger.warning(f"Company CIK not found for: {sanitized_company}")
            return {"error": "Company CIK not found"}
        
        # Get company CIK
        cik = company_search["cik"]
        logger.info(f"Found CIK: {cik} for company: {sanitized_company}")
        
        # Get the most recent 10-K filing
        filings = get_company_filings(cik, filing_type="10-K", limit=1)
        
        if isinstance(filings, dict) and "error" in filings:
            logger.warning(f"Error getting company filings: {filings['error']}")
            return {"error": filings["error"]}
        
        if not filings:
            logger.warning(f"No 10-K filings found for company with CIK: {cik}")
            # Try to get any filing type as a fallback
            filings = get_company_filings(cik, limit=1)
            if not filings or (isinstance(filings, dict) and "error" in filings):
                logger.warning(f"No filings found for company with CIK: {cik}")
                # Generate a basic SWOT with company information only
                return {"swot": f"# SWOT Analysis for {sanitized_company}\n\nInsufficient SEC filing data available to generate a detailed SWOT analysis. Please try another company or check back later."}
        
        logger.info(f"Found filing: {filings[0]['form']} from {filings[0]['filingDate']} for CIK: {cik}")
        
        # Get the content of the filing
        filing_content = get_filing_content(cik, filings[0]['accessionNumber'], filings[0]['primaryDocument'])
        
        if not filing_content or filing_content.startswith("Failed to get filing content"):
            logger.warning(f"Failed to get filing content for CIK: {cik}, Accession: {filings[0]['accessionNumber']}")
            return {"swot": f"# SWOT Analysis for {sanitized_company}\n\nUnable to retrieve filing content from SEC EDGAR. This may be due to temporary SEC API limitations or the filing format. Please try again later or try another company."}
        
        logger.info(f"Successfully retrieved filing content for CIK: {cik}, length: {len(filing_content)} characters")
        
        # Extract key sections from the filing
        business_section = extract_section(filing_content, "Item 1", "Item 1A")
        if not business_section:
            # Try alternative markers
            business_section = extract_section(filing_content, "ITEM 1", "ITEM 1A")
            if not business_section:
                business_section = extract_section(filing_content, "Business", "Risk Factors")
                if not business_section:
                    logger.warning(f"Could not extract business section for CIK: {cik}")
                    business_section = ""
        
        risk_factors = extract_section(filing_content, "Item 1A", "Item 1B")
        if not risk_factors:
            # Try alternative markers
            risk_factors = extract_section(filing_content, "ITEM 1A", "ITEM 1B")
            if not risk_factors:
                risk_factors = extract_section(filing_content, "Risk Factors", "Unresolved Staff Comments")
                if not risk_factors:
                    logger.warning(f"Could not extract risk factors for CIK: {cik}")
                    risk_factors = ""
        
        md_and_a = extract_section(filing_content, "Item 7", "Item 7A")
        if not md_and_a:
            # Try alternative markers
            md_and_a = extract_section(filing_content, "ITEM 7", "ITEM 7A")
            if not md_and_a:
                md_and_a = extract_section(filing_content, "Management's Discussion", "Quantitative and Qualitative Disclosures")
                if not md_and_a:
                    logger.warning(f"Could not extract MD&A for CIK: {cik}")
                    md_and_a = ""
        
        logger.info(f"Extracted sections - Business: {len(business_section)} chars, Risk Factors: {len(risk_factors)} chars, MD&A: {len(md_and_a)} chars")
        
        # Extract financial data
        financial_data = extract_financial_data(cik)
        
        # Generate SWOT analysis
        swot = f"# SWOT Analysis for {sanitized_company} based on SEC Filings\n\n"
        swot += f"*Based on {filings[0]['form']} filed on {filings[0]['filingDate']}*\n\n"
        
        # Strengths
        swot += "## Strengths\n\n"
        
        # Look for positive statements in business section and MD&A
        positive_indicators = [
            "market leader", "competitive advantage", "strong", "growth", "increase", 
            "profitable", "innovation", "patent", "proprietary", "exclusive", "success",
            "leading", "largest", "first", "best", "unique", "superior", "efficient"
        ]
        
        strengths_found = []
        
        # First try to find complete sentences containing the indicators
        for indicator in positive_indicators:
            # Search in business section
            if business_section:
                matches = re.finditer(r'([^.!?]*\b' + re.escape(indicator) + r'\b[^.!?]*[.!?])', business_section, re.IGNORECASE)
                for match in matches:
                    strength = match.group(1).strip()
                    if strength and len(strength) > 20 and strength not in strengths_found:
                        strengths_found.append(strength)
                        if len(strengths_found) >= 5:
                            break
            
            # If we have enough strengths, break
            if len(strengths_found) >= 5:
                break
            
            # Search in MD&A
            if md_and_a:
                matches = re.finditer(r'([^.!?]*\b' + re.escape(indicator) + r'\b[^.!?]*[.!?])', md_and_a, re.IGNORECASE)
                for match in matches:
                    strength = match.group(1).strip()
                    if strength and len(strength) > 20 and strength not in strengths_found:
                        strengths_found.append(strength)
                        if len(strengths_found) >= 5:
                            break
            
            # If we have enough strengths, break
            if len(strengths_found) >= 5:
                break
        
        # Add financial strengths if available
        if isinstance(financial_data, dict) and "error" not in financial_data:
            # Revenue growth
            if financial_data.get("revenue") and len(financial_data["revenue"]) >= 2:
                latest = financial_data["revenue"][0]
                previous = financial_data["revenue"][1]
                
                try:
                    change = ((latest["value"] - previous["value"]) / previous["value"]) * 100
                    
                    if change > 0:
                        strengths_found.append(f"Revenue increased by {change:.2f}% from {previous['date']} to {latest['date']}.")
                except (TypeError, ZeroDivisionError) as e:
                    logger.warning(f"Error calculating revenue change: {str(e)}")
            
            # Net Income growth
            if financial_data.get("netIncome") and len(financial_data["netIncome"]) >= 2:
                latest = financial_data["netIncome"][0]
                previous = financial_data["netIncome"][1]
                
                try:
                    if previous["value"] != 0:
                        change = ((latest["value"] - previous["value"]) / abs(previous["value"])) * 100
                        
                        if change > 0:
                            strengths_found.append(f"Net Income increased by {change:.2f}% from {previous['date']} to {latest['date']}.")
                except (TypeError, ZeroDivisionError) as e:
                    logger.warning(f"Error calculating net income change: {str(e)}")
        
        # Add strengths to SWOT
        if strengths_found:
            for strength in strengths_found[:5]:
                swot += f"- {strength}\n"
        else:
            # Generate generic strengths based on industry if no specific strengths found
            if "sicDescription" in company_search:
                swot += f"- Industry presence in {company_search['sicDescription']}.\n"
            swot += "- No additional specific strengths identified in recent SEC filings.\n"
        
        # Weaknesses
        swot += "\n## Weaknesses\n\n"
        
        # Extract weaknesses from risk factors
        weaknesses_found = []
        
        # Look for specific risk statements
        if risk_factors:
            risk_statements = re.finditer(r'([^.!?]*(?:risk|challenge|weakness|difficulty|problem|issue|decline|decrease|reduction)[^.!?]*[.!?])', risk_factors, re.IGNORECASE)
            
            for match in risk_statements:
                weakness = match.group(1).strip()
                if weakness and len(weakness) > 20 and weakness not in weaknesses_found:
                    weaknesses_found.append(weakness)
                    if len(weaknesses_found) >= 5:
                        break
        
        # Add financial weaknesses if available
        if isinstance(financial_data, dict) and "error" not in financial_data:
            # Revenue decline
            if financial_data.get("revenue") and len(financial_data["revenue"]) >= 2:
                latest = financial_data["revenue"][0]
                previous = financial_data["revenue"][1]
                
                try:
                    change = ((latest["value"] - previous["value"]) / previous["value"]) * 100
                    
                    if change < 0:
                        weaknesses_found.append(f"Revenue decreased by {abs(change):.2f}% from {previous['date']} to {latest['date']}.")
                except (TypeError, ZeroDivisionError) as e:
                    logger.warning(f"Error calculating revenue change: {str(e)}")
            
            # Net Income decline
            if financial_data.get("netIncome") and len(financial_data["netIncome"]) >= 2:
                latest = financial_data["netIncome"][0]
                previous = financial_data["netIncome"][1]
                
                try:
                    if previous["value"] != 0:
                        change = ((latest["value"] - previous["value"]) / abs(previous["value"])) * 100
                        
                        if change < 0:
                            weaknesses_found.append(f"Net Income decreased by {abs(change):.2f}% from {previous['date']} to {latest['date']}.")
                except (TypeError, ZeroDivisionError) as e:
                    logger.warning(f"Error calculating net income change: {str(e)}")
        
        # Add weaknesses to SWOT
        if weaknesses_found:
            for weakness in weaknesses_found[:5]:
                swot += f"- {weakness}\n"
        else:
            swot += "- No specific weaknesses identified in recent SEC filings.\n"
            # Add a generic weakness based on industry if available
            if "sicDescription" in company_search:
                swot += f"- Potential exposure to general risks associated with the {company_search['sicDescription']} industry.\n"
        
        # Opportunities
        swot += "\n## Opportunities\n\n"
        
        # Look for opportunity statements
        opportunity_indicators = [
            "opportunity", "potential", "growth", "expansion", "new market", 
            "emerging", "development", "future", "prospect", "strategy",
            "innovation", "technology", "digital", "transform", "invest"
        ]
        
        opportunities_found = []
        
        for indicator in opportunity_indicators:
            # Search in business section
            if business_section:
                matches = re.finditer(r'([^.!?]*\b' + re.escape(indicator) + r'\b[^.!?]*[.!?])', business_section, re.IGNORECASE)
                for match in matches:
                    opportunity = match.group(1).strip()
                    if opportunity and len(opportunity) > 20 and opportunity not in opportunities_found:
                        opportunities_found.append(opportunity)
                        if len(opportunities_found) >= 5:
                            break
            
            # If we have enough opportunities, break
            if len(opportunities_found) >= 5:
                break
            
            # Search in MD&A
            if md_and_a:
                matches = re.finditer(r'([^.!?]*\b' + re.escape(indicator) + r'\b[^.!?]*[.!?])', md_and_a, re.IGNORECASE)
                for match in matches:
                    opportunity = match.group(1).strip()
                    if opportunity and len(opportunity) > 20 and opportunity not in opportunities_found:
                        opportunities_found.append(opportunity)
                        if len(opportunities_found) >= 5:
                            break
            
            # If we have enough opportunities, break
            if len(opportunities_found) >= 5:
                break
        
        # Add opportunities to SWOT
        if opportunities_found:
            for opportunity in opportunities_found[:5]:
                swot += f"- {opportunity}\n"
        else:
            swot += "- No specific opportunities identified in recent SEC filings.\n"
            # Add generic opportunities
            swot += "- Potential for industry growth and market expansion.\n"
            if "sicDescription" in company_search:
                swot += f"- Possible innovation opportunities in the {company_search['sicDescription']} sector.\n"
        
        # Threats
        swot += "\n## Threats\n\n"
        
        # Extract threats from risk factors
        threats_found = []
        
        # Look for specific threat statements
        if risk_factors:
            threat_statements = re.finditer(r'([^.!?]*(?:competition|competitor|threat|risk|regulatory|regulation|law|litigation|lawsuit|conflict|dispute)[^.!?]*[.!?])', risk_factors, re.IGNORECASE)
            
            for match in threat_statements:
                threat = match.group(1).strip()
                if threat and len(threat) > 20 and threat not in threats_found:
                    threats_found.append(threat)
                    if len(threats_found) >= 5:
                        break
        
        # Add threats to SWOT
        if threats_found:
            for threat in threats_found[:5]:
                swot += f"- {threat}\n"
        else:
            swot += "- No specific threats identified in recent SEC filings.\n"
            # Add generic threats
            swot += "- General market competition and industry challenges.\n"
            swot += "- Potential regulatory changes affecting business operations.\n"
        
        logger.info(f"Successfully generated SWOT analysis for {sanitized_company}")
        return {"swot": swot}
    
    except Exception as e:
        logger.error(f"Error generating SWOT: {str(e)}")
        # Return a basic SWOT analysis instead of an error to ensure something is displayed
        swot = f"# SWOT Analysis for {sanitized_company}\n\n"
        swot += "*Note: This is a simplified analysis due to data retrieval limitations.*\n\n"
        
        swot += "## Strengths\n\n"
        swot += "- Company has established presence in its industry.\n"
        swot += "- Registered with SEC, indicating compliance with financial reporting requirements.\n\n"
        
        swot += "## Weaknesses\n\n"
        swot += "- Detailed financial analysis not available at this time.\n"
        swot += "- Limited public information for comprehensive assessment.\n\n"
        
        swot += "## Opportunities\n\n"
        swot += "- Potential for growth in current and new markets.\n"
        swot += "- Possibilities for strategic partnerships and acquisitions.\n\n"
        
        swot += "## Threats\n\n"
        swot += "- Competitive pressures within the industry.\n"
        swot += "- Regulatory changes that may impact operations.\n"
        swot += "- Economic fluctuations affecting market conditions.\n"
        
        return {"swot": swot}
def process_user_query(query, company_context=None):
    """Process general user queries about a company using SEC data"""
    sanitized_query = sanitize_input(query)
    
    if not company_context:
        return "Please select a company first by entering a company name in the sidebar and clicking 'Analyze Company'."
    
    try:
        # Search for the company
        company_search = search_company(company_context)
        
        if "error" in company_search:
            return f"Error: {company_search['error']}"
        
        if "cik" not in company_search:
            return "Error: Company CIK not found"
        
        # Get company CIK
        cik = company_search["cik"]
        
        # Process different types of queries
        query_lower = sanitized_query.lower()
        
        # Financial data query
        if any(term in query_lower for term in ["financial", "revenue", "income", "profit", "loss", "earnings", "assets", "liabilities"]):
            financial_data = extract_financial_data(cik)
            
            if isinstance(financial_data, dict) and "error" in financial_data:
                return f"Error retrieving financial data: {financial_data['error']}"
            
            response = f"Financial data for {company_context} from SEC filings:\n\n"
            
            # Revenue
            if financial_data["revenue"]:
                response += "## Revenue\n\n"
                for item in financial_data["revenue"][:3]:  # Show last 3 years
                    response += f"- {item['date']}: ${item['value']:,.2f}\n"
            
            # Net Income
            if financial_data["netIncome"]:
                response += "\n## Net Income\n\n"
                for item in financial_data["netIncome"][:3]:  # Show last 3 years
                    response += f"- {item['date']}: ${item['value']:,.2f}\n"
            
            # Total Assets
            if financial_data["totalAssets"]:
                response += "\n## Total Assets\n\n"
                for item in financial_data["totalAssets"][:3]:  # Show last 3 years
                    response += f"- {item['date']}: ${item['value']:,.2f}\n"
            
            # Total Liabilities
            if financial_data["totalLiabilities"]:
                response += "\n## Total Liabilities\n\n"
                for item in financial_data["totalLiabilities"][:3]:  # Show last 3 years
                    response += f"- {item['date']}: ${item['value']:,.2f}\n"
            
            return response
        
        # Filings query
        elif any(term in query_lower for term in ["filing", "report", "10-k", "10-q", "8-k", "sec"]):
            filings = get_company_filings(cik, limit=10)
            
            if isinstance(filings, dict) and "error" in filings:
                return f"Error retrieving filings: {filings['error']}"
            
            response = f"Recent SEC filings for {company_context}:\n\n"
            
            for filing in filings:
                response += f"- {filing['form']} filed on {filing['filingDate']}\n"
                response += f"  Accession Number: {filing['accessionNumber']}\n"
                if filing.get('reportDate'):
                    response += f"  Report Date: {filing['reportDate']}\n"
                response += f"  Document: {filing['primaryDocument']}\n\n"
            
            return response
        
        # Risk factors query
        elif any(term in query_lower for term in ["risk", "risks", "risk factors"]):
            # Get the most recent 10-K filing
            filings = get_company_filings(cik, filing_type="10-K", limit=1)
            
            if isinstance(filings, dict) and "error" in filings:
                return f"Error retrieving filings: {filings['error']}"
            
            if not filings:
                return "No 10-K filings found to extract risk factors."
            
            # Get the content of the 10-K filing
            filing_content = get_filing_content(cik, filings[0]['accessionNumber'], filings[0]['primaryDocument'])
            
            # Extract risk factors section
            risk_factors = extract_section(filing_content, "Item 1A", "Item 1B")
            
            if not risk_factors:
                return "Could not extract risk factors from the 10-K filing."
            
            # Limit the response length
            if len(risk_factors) > 4000:
                risk_factors = risk_factors[:4000] + "... (truncated)"
            
            response = f"Risk Factors for {company_context} from the most recent 10-K filing ({filings[0]['filingDate']}):\n\n"
            response += risk_factors
            
            return response
        
        # Business description query
        elif any(term in query_lower for term in ["business", "company", "description", "what does", "what is"]):
            # Get the most recent 10-K filing
            filings = get_company_filings(cik, filing_type="10-K", limit=1)
            
            if isinstance(filings, dict) and "error" in filings:
                return f"Error retrieving filings: {filings['error']}"
            
            if not filings:
                return "No 10-K filings found to extract business description."
            
            # Get the content of the 10-K filing
            filing_content = get_filing_content(cik, filings[0]['accessionNumber'], filings[0]['primaryDocument'])
            
            # Extract business section
            business_section = extract_section(filing_content, "Item 1", "Item 1A")
            
            if not business_section:
                return "Could not extract business description from the 10-K filing."
            
            # Limit the response length
            if len(business_section) > 4000:
                business_section = business_section[:4000] + "... (truncated)"
            
            response = f"Business Description for {company_context} from the most recent 10-K filing ({filings[0]['filingDate']}):\n\n"
            response += business_section
            
            return response
        
        # Default response
        else:
            return f"I can provide information about {company_context} based on SEC filings. Try asking about financial data, recent filings, risk factors, or business description."
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"Error processing your query: {str(e)}"

# Sidebar for settings and company selection
with st.sidebar:
    st.title("Company Deep Dive")
    st.markdown("---")
    
    company_name = st.text_input("Enter a company name or ticker:", key="company_input")
    
    if st.button("Analyze Company") and company_name:
        sanitized_company = sanitize_input(company_name)
        if validate_company_name(sanitized_company):
            with st.spinner(f"Analyzing {sanitized_company} using SEC EDGAR data..."):
                # Search for the company
                company_search = search_company(sanitized_company)
                
                if "error" in company_search:
                    st.error(f"Error: {company_search['error']}")
                elif "cik" not in company_search:
                    st.error("Company CIK not found")
                else:
                    # Get company CIK
                    cik = company_search["cik"]
                    
                    # Fetch company information
                    company_info = fetch_company_info(sanitized_company)
                    sentiment_result = analyze_company_sentiment(sanitized_company)
                    swot_result = get_company_swot(sanitized_company)
                    
                    # Store results in session state
                    st.session_state.company_data = {
                        "name": sanitized_company,
                        "cik": cik,
                        "info": company_info.get("info", "Information not available"),
                        "sentiment": sentiment_result.get("sentiment", "Sentiment analysis not available"),
                        "swot": swot_result.get("swot", "SWOT analysis not available"),
                        "financials": company_info.get("financials", {})
                    }
                    
                    # Add system message to chat
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"I've analyzed {sanitized_company} using SEC EDGAR data. You can ask me specific questions about this company now."
                    })
        else:
            st.error("Please enter a valid company name or ticker")
    
    st.markdown("---")
    st.markdown("### Analysis Options")
    
    if st.session_state.company_data and "name" in st.session_state.company_data:
        if st.button("Market Sentiment"):
            st.session_state.messages.append({
                "role": "user", 
                "content": f"What's the market sentiment for {st.session_state.company_data['name']}?"
            })
            st.session_state.messages.append({
                "role": "assistant", 
                "content": st.session_state.company_data["sentiment"]
            })
            
        if st.button("SWOT Analysis"):
            st.session_state.messages.append({
                "role": "user", 
                "content": f"Provide a SWOT analysis for {st.session_state.company_data['name']}"
            })
            st.session_state.messages.append({
                "role": "assistant", 
                "content": st.session_state.company_data["swot"]
            })
            
        if st.button("Recent SEC Filings"):
            st.session_state.messages.append({
                "role": "user", 
                "content": f"Show me recent SEC filings for {st.session_state.company_data['name']}"
            })
            
            if "cik" in st.session_state.company_data:
                cik = st.session_state.company_data["cik"]
                filings = get_company_filings(cik, limit=10)
                
                if isinstance(filings, dict) and "error" in filings:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"Error retrieving filings: {filings['error']}"
                    })
                else:
                    filings_overview = f"# Recent SEC Filings for {st.session_state.company_data['name']}\n\n"
                    
                    for filing in filings:
                        filings_overview += f"- **{filing['form']}** filed on {filing['filingDate']}\n"
                        if filing.get('reportDate'):
                            filings_overview += f"  Report Date: {filing['reportDate']}\n"
                        filings_overview += f"  Document: {filing['primaryDocument']}\n\n"
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": filings_overview
                    })
            else:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "Company CIK not available to retrieve SEC filings."
                })
        
        if st.expander("Earnings Call Transcript"):
            # Create columns for year and quarter selection
            col1, col2 = st.columns(2)
            url = "https://www.sec.gov/files/company_tickers.json" 
            response = requests.get(url, headers={'User-Agent': 'your-email@example.com'}) 
            companies = pd.DataFrame.from_dict(response.json(), orient='index') # Search for company 
            result = companies[companies['title'].str.contains(company_name, case=False)]
            ticker=result['ticker'].values[0]
            df=yf.Ticker(ticker).earnings_dates.reset_index() 
            df.columns = ['Earnings Date'] + list(df.columns[1:]) 
            df['Year'] = df['Earnings Date'].dt.year 
            df['Quarter'] = df['Earnings Date'].dt.quarter
            with col1:
                # Default to current year
                current_year = datetime.datetime.now().year
                year_options = sorted(set(df['Year']), reverse=True)
                selected_year = st.selectbox("Year:", year_options, key="transcript_year")
            
            with col2: # Filter quarters based on selected year 
                quarters_for_year = sorted(df[df['Year'] == selected_year]['Quarter'].unique(), reverse=False) # Default to most recent quarter for that year 
                current_month = datetime.datetime.now().month 
                default_quarter = ((current_month - 1) // 3) + 1 
                if default_quarter not in quarters_for_year: 
                    default_quarter = quarters_for_year[-1]  # fallback to latest available quarter 
                selected_quarter = st.selectbox( "Quarter:", quarters_for_year, index=quarters_for_year.index(default_quarter), key="transcript_quarter" )
            
            # Separate button outside of any nested conditions
            if st.button("Fetch Transcript", key="fetch_transcript"):
                st.session_state.messages.append({
                    "role": "user", 
                    "content": f"Show me the earnings call transcript for {st.session_state.company_data['name']} (Year: {selected_year}, Quarter: {selected_quarter})"
                })
                
                # Create a placeholder to show status directly in the expander
                status_placeholder = st.empty()
                status_placeholder.info(f"Fetching earnings transcript for {st.session_state.company_data['name']}...")
                
                try:
                    # Get the transcript data
                    transcript_result = get_earnings_transcript(st.session_state.company_data['name'], selected_year, selected_quarter) 
                    document = Document(page_content=str(transcript_result), metadata={ "company": str(st.session_state.company_data["name"]), "year": str(selected_year), "quarter": str(selected_quarter), "source": "motley_fool" }) 
                    rag.vector_store.add_documents([document])
                    analysis = rag.query(f"Get analysis for {st.session_state.company_data['name']} " f"with year {selected_year} and quarter {selected_quarter}", lookback_hours=24)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": transcript_result+"\n"+analysis["answer"]
                    })
                    
                    # Force a rerun to update the chat display
                    st.rerun()
                except Exception as e:
                    error_msg = f"An error occurred while fetching the transcript: {str(e)}"
                    status_placeholder.error(error_msg)
                    logger.error(error_msg)
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This Company Deep Dive Chatbot helps you analyze companies using official SEC EDGAR data.
    
    Enter a company name or ticker in the sidebar and click 'Analyze Company' to get started.
    """)

# Main chat interface
st.title("Company Deep Dive Chatbot 🏢")
st.caption("Powered by SEC EDGAR data")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about a company..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate a response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            company_context = st.session_state.company_data.get("name", None)
            response = process_user_query(prompt, company_context)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Instructions at the bottom
st.markdown("---")
st.markdown("""
### How to use this chatbot:
1. Enter a company name or ticker in the sidebar and click "Analyze Company"
2. Use the analysis buttons in the sidebar to get specific information
3. Or simply ask questions in the chat input below
""")

# Footer with data source notice
st.markdown("---")
st.caption("""
This application uses data from the SEC EDGAR database, which contains official company filings.
Data is retrieved in real-time from SEC.gov and is subject to their terms of service.
""")
