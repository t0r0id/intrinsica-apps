#!/usr/bin/env python3
"""
Streamlit Radar Tracker App
Display Key Issues and Hypotheses for companies
"""

import streamlit as st
from datetime import date, timedelta
from typing import List, Dict, Any, Tuple
import string
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sqlalchemy.exc import SQLAlchemyError
from utils.auth import check_authentication
from utils.database import (
    get_database_connection, 
    Company, 
    Transcript, 
    ConferenceInsightsReport, 
    TranscriptKeyIssue,
    HypothesisState,
    Hypothesis,
    TranscriptKeyDevelopment,
    PublishedRadarReport,
    func
)



# Check authentication
if not check_authentication():
    st.error("Please login from the main page to access this application.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Intrinsica Report Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data"""
    nltk.data.path.append("./data/nltk")
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

# Initialize NLTK
download_nltk_data()

# Initialize preprocessing tools
@st.cache_resource
def get_preprocessing_tools():
    """Get NLTK preprocessing tools"""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    return lemmatizer, stop_words

# Data fetching functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_companies():
    """Fetch all active companies"""
    SessionLocal = get_database_connection()
    session = SessionLocal()
    
    try:
        companies = session.query(Company).filter(Company.is_active == True).order_by(Company.name).all()
        return [(c.id, c.name, c.ticker) for c in companies]
    except SQLAlchemyError as e:
        st.error(f"Error fetching companies: {str(e)}")
        return []
    finally:
        session.close()

@st.cache_data(ttl=300)
def get_key_issues(company_id: int, start_date: date, end_date: date) -> List[Dict[str, Any]]:
    """Fetch key issues for a company within date range"""
    SessionLocal = get_database_connection()
    session = SessionLocal()
    
    try:
        query = (
            session.query(
                TranscriptKeyIssue.id,
                TranscriptKeyIssue.title,
                TranscriptKeyIssue.analyst_concern,
                TranscriptKeyIssue.management_response,
                TranscriptKeyIssue.counterpoints,
                TranscriptKeyIssue.followup_questions,
                TranscriptKeyIssue.created_at,
                Transcript.id.label('transcript_id'),
                Transcript.published_date,
                Transcript.title.label('transcript_title'),
                Transcript.quarter
            )
            .join(ConferenceInsightsReport, TranscriptKeyIssue.conference_insights_report_id == ConferenceInsightsReport.id)
            .join(Transcript, ConferenceInsightsReport.transcript_id == Transcript.id)
            .join(Company, Transcript.company_id == Company.id)
            .filter(
                Company.id == company_id,
                ConferenceInsightsReport.is_active == True,
                Transcript.published_date >= start_date,
                Transcript.published_date <= end_date
            )
            .order_by(Transcript.published_date.desc())
        )
        
        results = query.all()
        
        return [
            {
                'id': r.id,
                'title': r.title,
                'analyst_concern': r.analyst_concern,
                'management_response': r.management_response,
                'counterpoints': r.counterpoints or [],
                'followup_questions': r.followup_questions or [],
                'transcript_id': r.transcript_id,
                'published_date': r.published_date,
                'transcript_title': r.transcript_title,
                'quarter': r.quarter
            }
            for r in results
        ]
        
    except SQLAlchemyError as e:
        st.error(f"Error fetching key issues: {str(e)}")
        return []
    finally:
        session.close()

@st.cache_data(ttl=300)
def get_hypotheses(company_id: int, start_date: date, end_date: date) -> List[Dict[str, Any]]:
    """Fetch hypotheses for a company within date range"""
    SessionLocal = get_database_connection()
    session = SessionLocal()
    
    try:
        query = (
            session.query(
                Hypothesis.id,
                Hypothesis.title,
                Hypothesis.reasoning,
                Hypothesis.created_at,
                Hypothesis.last_updated_at,
                TranscriptKeyDevelopment.created_at.label('development_date'),
                PublishedRadarReport.id.label('radar_report_id'),
                PublishedRadarReport.title.label('radar_report_title')
            )
            .join(HypothesisState, Hypothesis.hypothesis_state_id == HypothesisState.id)
            .join(TranscriptKeyDevelopment, HypothesisState.source_key_development_id == TranscriptKeyDevelopment.id)
            .outerjoin(
                PublishedRadarReport,
                (PublishedRadarReport.target_company_id == company_id) &
                (func.date(PublishedRadarReport.published_date) == func.date(Hypothesis.last_updated_at))
            )
            .filter(
                Hypothesis.target_company_id == company_id,
                Hypothesis.created_at >= start_date,
                Hypothesis.created_at <= end_date
            )
            .order_by(Hypothesis.created_at.desc())
        )
        
        results = query.all()
        
        return [
            {
                'id': r.id,
                'title': r.title,
                'reasoning': r.reasoning,
                'created_at': r.created_at,
                'last_updated_at': r.last_updated_at,
                'development_date': r.development_date,
                'radar_report_id': r.radar_report_id,
                'radar_report_title': r.radar_report_title
            }
            for r in results
        ]
        
    except SQLAlchemyError as e:
        st.error(f"Error fetching hypotheses: {str(e)}")
        return []
    finally:
        session.close()

@st.cache_data(ttl=300)
def get_key_developments(company_id: int, start_date: date, end_date: date) -> List[Dict[str, Any]]:
    """Fetch key developments for a company within date range"""
    SessionLocal = get_database_connection()
    session = SessionLocal()
    
    try:
        query = (
            session.query(
                TranscriptKeyDevelopment.id,
                TranscriptKeyDevelopment.subject,
                TranscriptKeyDevelopment.title,
                TranscriptKeyDevelopment.description,
                TranscriptKeyDevelopment.category,
                TranscriptKeyDevelopment.created_at,
                Transcript.id.label('transcript_id'),
                Transcript.published_date,
                Transcript.title.label('transcript_title'),
                Transcript.quarter
            )
            .join(ConferenceInsightsReport, TranscriptKeyDevelopment.conference_insights_report_id == ConferenceInsightsReport.id)
            .join(Transcript, ConferenceInsightsReport.transcript_id == Transcript.id)
            .join(Company, Transcript.company_id == Company.id)
            .filter(
                Company.id == company_id,
                ConferenceInsightsReport.is_active == True,
                Transcript.published_date >= start_date,
                Transcript.published_date <= end_date
            )
            .order_by(Transcript.published_date.desc())
        )
        
        results = query.all()
        
        return [
            {
                'id': r.id,
                'subject': r.subject,
                'title': r.title,
                'description': r.description,
                'category': r.category,
                'transcript_id': r.transcript_id,
                'published_date': r.published_date,
                'transcript_title': r.transcript_title,
                'quarter': r.quarter
            }
            for r in results
        ]
        
    except SQLAlchemyError as e:
        st.error(f"Error fetching key developments: {str(e)}")
        return []
    finally:
        session.close()

# BM25 search implementation using rank-bm25
def preprocess_text(text: str) -> List[str]:
    """
    Preprocess text for BM25 indexing with:
    - Case normalization
    - Punctuation removal 
    - Tokenization
    - Stopword removal
    - Lemmatization
    """
    if not text:
        return []
    
    # Get preprocessing tools
    lemmatizer, stop_words = get_preprocessing_tools()
    
    # Case normalization
    text = text.lower()
    
    # Remove punctuation and symbols
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize using NLTK
    tokens = word_tokenize(text)
    
    # Remove stopwords and apply lemmatization
    processed_tokens = []
    for token in tokens:
        # Skip if stopword or not alphabetic
        if token not in stop_words and token.isalpha() and len(token) > 2:
            # Apply lemmatization
            lemmatized = lemmatizer.lemmatize(token)
            processed_tokens.append(lemmatized)
    
    return processed_tokens

def build_search_corpus(key_issues: List[Dict[str, Any]], hypotheses: List[Dict[str, Any]], key_developments: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]], BM25Okapi]:
    """Build BM25 corpus from key issues, hypotheses, and key developments"""
    corpus_texts = []
    corpus_items = []
    
    # Add key issues to corpus
    for issue in key_issues:
        searchable_text = f"{issue.get('title', '')} {issue.get('analyst_concern', '')} {issue.get('management_response', '')} {' '.join(issue.get('counterpoints', []))} {' '.join(issue.get('followup_questions', []))}"
        corpus_texts.append(searchable_text.strip())
        corpus_items.append({**issue, 'type': 'key_issue'})
    
    # Add hypotheses to corpus
    for hypothesis in hypotheses:
        searchable_text = f"{hypothesis.get('title', '')} {hypothesis.get('reasoning', '')}"
        corpus_texts.append(searchable_text.strip())
        corpus_items.append({**hypothesis, 'type': 'hypothesis'})
    
    # Add key developments to corpus
    for development in key_developments:
        searchable_text = f"{development.get('subject', '')} {development.get('title', '')} {development.get('description', '')} {development.get('category', '')}"
        corpus_texts.append(searchable_text.strip())
        corpus_items.append({**development, 'type': 'key_development'})
    
    # Preprocess corpus with full text processing pipeline
    tokenized_corpus = [preprocess_text(text) for text in corpus_texts]
    
    # Ensure we have at least some non-empty documents
    if not any(tokenized_corpus):
        # Create a dummy document to prevent BM25 from failing
        tokenized_corpus = [['dummy']]
        corpus_texts = ['dummy document']
        corpus_items = [{'type': 'dummy', 'title': 'No data'}]
    
    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    
    return corpus_texts, corpus_items, bm25

def rerank_with_bm25(key_issues: List[Dict[str, Any]], hypotheses: List[Dict[str, Any]], key_developments: List[Dict[str, Any]], query: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Rerank all key issues, hypotheses, and key developments using BM25 scores"""
    if not query.strip():
        return key_issues, hypotheses, key_developments
    
    if not key_issues and not hypotheses and not key_developments:
        return [], [], []
    
    # Build corpus and BM25 index
    _, corpus_items, bm25 = build_search_corpus(key_issues, hypotheses, key_developments)
    
    # Preprocess query with same pipeline as corpus
    query_tokens = preprocess_text(query)
    if not query_tokens:
        return key_issues, hypotheses, key_developments
    
    # Get BM25 scores for ALL documents
    scores = bm25.get_scores(query_tokens)
    
    # Add scores to all items and separate by type
    scored_key_issues = []
    scored_hypotheses = []
    scored_key_developments = []
    
    for item, score in zip(corpus_items, scores):
        item_copy = item.copy()
        item_copy['search_score'] = score  # Include ALL scores, even 0
        
        if item['type'] == 'key_issue':
            del item_copy['type']
            scored_key_issues.append(item_copy)
        elif item['type'] == 'hypothesis':
            del item_copy['type']
            scored_hypotheses.append(item_copy)
        else:  # key_development
            del item_copy['type']
            scored_key_developments.append(item_copy)
    
    # Sort ALL by score descending (highest relevance first)
    scored_key_issues.sort(key=lambda x: x['search_score'], reverse=True)
    scored_hypotheses.sort(key=lambda x: x['search_score'], reverse=True)
    scored_key_developments.sort(key=lambda x: x['search_score'], reverse=True)
    
    return scored_key_issues, scored_hypotheses, scored_key_developments

def main():
    """Main Streamlit application"""
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎯 Configuration")
        
        # Company selection
        st.subheader("Company Selection")
        companies = get_companies()
        
        if not companies:
            st.error("No companies found. Please check database connection.")
            st.stop()
        
        company_options = {f"{name} ({ticker})": (id, name, ticker) for id, name, ticker in companies}
        selected_company_display = st.selectbox(
            "Select Company",
            options=list(company_options.keys()),
            help="Choose a company to view their radar tracking data"
        )
        
        company_id, company_name, company_ticker = company_options[selected_company_display]
        
        # Date selection
        st.subheader("Date Ranges")
        
        # Transcript date range (for Key Issues and Key Developments)
        st.markdown("**Transcript Date Range**")
        transcript_col1, transcript_col2 = st.columns(2)
        
        with transcript_col1:
            transcript_start = st.date_input(
                "From",
                value=date.today() - timedelta(days=180),
                key="transcript_start",
                help="Start date for Key Issues and Key Developments"
            )
        
        with transcript_col2:
            transcript_end = st.date_input(
                "To",
                value=date.today(),
                key="transcript_end",
                help="End date for Key Issues and Key Developments"
            )
        
        # Hypotheses date range
        st.markdown("**Hypotheses Date Range**")
        hypotheses_col1, hypotheses_col2 = st.columns(2)
        
        with hypotheses_col1:
            hypotheses_start = st.date_input(
                "From",
                value=date.today() - timedelta(days=365),
                key="hypotheses_start",
                help="Start date for Hypotheses"
            )
        
        with hypotheses_col2:
            hypotheses_end = st.date_input(
                "To",
                value=date.today(),
                key="hypotheses_end",
                help="End date for Hypotheses"
            )
        
        # Validation
        if transcript_start > transcript_end:
            st.error("Transcript start date must be before end date")
        if hypotheses_start > hypotheses_end:
            st.error("Hypotheses start date must be before end date")
    
    # Main content area
    st.header(f"📊 {company_name} ({company_ticker})")
    
    # Check if date ranges are valid
    dates_valid = (transcript_start <= transcript_end) and (hypotheses_start <= hypotheses_end)
    
    if not dates_valid:
        st.warning("⚠️ Please fix the date range errors in the sidebar before viewing data.")
        st.stop()
    
    # Search bar
    st.subheader("🔍 Search")
    search_query = st.text_input(
        "Search in Key Issues, Hypotheses, and Key Developments",
        placeholder="e.g., tariff, revenue, guidance, AI, margin...",
        help="BM25 reranking: Shows ALL items ranked by relevance. Most relevant items appear first. Preprocessing includes lemmatization (tariffs→tariff, growing→growth)."
    )
    
    # Load data
    with st.spinner("Loading data..."):
        key_issues = get_key_issues(company_id, transcript_start, transcript_end)
        hypotheses = get_hypotheses(company_id, hypotheses_start, hypotheses_end)
        key_developments = get_key_developments(company_id, transcript_start, transcript_end)
    
    # Apply BM25 reranking
    if search_query.strip():
        key_issues, hypotheses, key_developments = rerank_with_bm25(key_issues, hypotheses, key_developments, search_query)
        
        # Count items with positive scores
        relevant_key_issues = sum(1 for item in key_issues if item.get('search_score', 0) > 0)
        relevant_hypotheses = sum(1 for item in hypotheses if item.get('search_score', 0) > 0)
        relevant_key_developments = sum(1 for item in key_developments if item.get('search_score', 0) > 0)
        
        if relevant_key_issues > 0 or relevant_hypotheses > 0 or relevant_key_developments > 0:
            st.success(f"🔍 Reranked by relevance for: **{search_query}** - {relevant_key_issues} relevant key issues, {relevant_hypotheses} relevant hypotheses, {relevant_key_developments} relevant key developments (BM25 scores)")
        else:
            st.info(f"🔍 No highly relevant matches for: **{search_query}**, but showing all {len(key_issues)} key issues, {len(hypotheses)} hypotheses, and {len(key_developments)} key developments ranked by relevance.")
            st.caption("💡 Items with score 0.0 have no matching terms but are still shown for completeness.")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Key Issues", "💡 Hypotheses", "📈 Key Developments"])
    
    with tab1:
        # Show current date range
        st.caption(f"📅 {transcript_start.strftime('%Y-%m-%d')} to {transcript_end.strftime('%Y-%m-%d')}")
        
        if key_issues:
            if search_query.strip():
                relevant_count = sum(1 for item in key_issues if item.get('search_score', 0) > 0)
                st.success(f"Showing {len(key_issues)} key issues ranked by relevance ({relevant_count} with positive scores)")
            else:
                st.success(f"Found {len(key_issues)} key issues")
            
            for issue in key_issues:
                issue_date = issue['published_date'].strftime('%Y-%m-%d')
                issue_url = f"https://app.intrinsica.ai/conference_insights/{issue['transcript_id']}/?element_id=issue-{issue['id']}"
                issue_title = issue['title'] or 'Untitled'
                
                # Show search score if available
                score_text = ""
                if 'search_score' in issue:
                    score = issue['search_score']
                    if score > 0:
                        score_text = f" (Relevance: {score:.2f})"
                    else:
                        score_text = f" (No match: {score:.2f})"
                
                with st.expander(f"**{issue_title}** [{issue_date}]{score_text}"):
                    # Add hyperlink at the top of the expanded content
                    st.markdown(f"🔗 [**View in Conference Insights**]({issue_url})")
                    st.markdown("---")
                    
                    st.markdown(f"**Transcript**: {issue['transcript_title']}")
                    st.markdown(f"**Quarter**: {issue['quarter']}")
                    
                    if issue['analyst_concern']:
                        st.markdown("**Analyst Concern:**")
                        st.write(issue['analyst_concern'])
                    
                    if issue['management_response']:
                        st.markdown("**Management Response:**")
                        st.write(issue['management_response'])
                    
                    if issue['counterpoints']:
                        st.markdown("**Counterpoints:**")
                        for point in issue['counterpoints']:
                            st.write(f"• {point}")
                    
                    if issue['followup_questions']:
                        st.markdown("**Follow-up Questions:**")
                        for question in issue['followup_questions']:
                            st.write(f"• {question}")
        else:
            if search_query.strip():
                st.info("No key issues found matching your search criteria")
            else:
                st.info("No key issues found for the selected time period")
    
    with tab2:
        # Show current date range
        st.caption(f"📅 {hypotheses_start.strftime('%Y-%m-%d')} to {hypotheses_end.strftime('%Y-%m-%d')}")
        
        if hypotheses:
            if search_query.strip():
                relevant_count = sum(1 for item in hypotheses if item.get('search_score', 0) > 0)
                st.success(f"Showing {len(hypotheses)} hypotheses ranked by relevance ({relevant_count} with positive scores)")
            else:
                st.success(f"Found {len(hypotheses)} hypotheses")
            
            for hypothesis in hypotheses:
                hyp_date = hypothesis['created_at'].strftime('%Y-%m-%d')
                hyp_title = hypothesis['title'] or 'Untitled'
                
                # Show search score if available
                score_text = ""
                if 'search_score' in hypothesis:
                    score = hypothesis['search_score']
                    if score > 0:
                        score_text = f" (Relevance: {score:.2f})"
                    else:
                        score_text = f" (No match: {score:.2f})"
                
                with st.expander(f"**{hyp_title}** [{hyp_date}]{score_text}"):
                    # Add radar report link at the top if available
                    if hypothesis['radar_report_id']:
                        radar_url = f"https://app.intrinsica.ai/radar_report/{hypothesis['radar_report_id']}"
                        if hypothesis['radar_report_title']:
                            st.markdown(f"📡 [**View Radar Report: {hypothesis['radar_report_title']}**]({radar_url})")
                        else:
                            st.markdown(f"📡 [**View Radar Report**]({radar_url})")
                        
                        # Show last updated date for context
                        if hypothesis['last_updated_at']:
                            updated_date = hypothesis['last_updated_at'].strftime('%Y-%m-%d')
                            st.caption(f"Report published: {updated_date}")
                        st.markdown("---")
                    
                    if hypothesis['reasoning']:
                        st.markdown("**Reasoning:**")
                        st.write(hypothesis['reasoning'])
                    
                    if hypothesis['development_date']:
                        dev_date = hypothesis['development_date'].strftime('%Y-%m-%d')
                        st.markdown(f"**Development Date**: {dev_date}")
        else:
            if search_query.strip():
                st.info("No hypotheses found matching your search criteria")
            else:
                st.info("No hypotheses found for the selected time period")
    
    with tab3:
        # Show current date range
        st.caption(f"📅 {transcript_start.strftime('%Y-%m-%d')} to {transcript_end.strftime('%Y-%m-%d')}")
        
        if key_developments:
            if search_query.strip():
                relevant_count = sum(1 for item in key_developments if item.get('search_score', 0) > 0)
                st.success(f"Showing {len(key_developments)} key developments ranked by relevance ({relevant_count} with positive scores)")
            else:
                st.success(f"Found {len(key_developments)} key developments")
            
            for development in key_developments:
                dev_date = development['published_date'].strftime('%Y-%m-%d')
                dev_url = f"https://app.intrinsica.ai/conference_insights/{development['transcript_id']}/?element_id=development-{development['id']}"
                dev_title = development['title'] or development['subject'] or 'Untitled'
                
                # Show search score if available
                score_text = ""
                if 'search_score' in development:
                    score = development['search_score']
                    if score > 0:
                        score_text = f" (Relevance: {score:.2f})"
                    else:
                        score_text = f" (No match: {score:.2f})"
                
                with st.expander(f"**{dev_title}** [{dev_date}]{score_text}"):
                    # Add hyperlink at the top of the expanded content
                    st.markdown(f"🔗 [**View in Conference Insights**]({dev_url})")
                    st.markdown("---")
                    
                    st.markdown(f"**Transcript**: {development['transcript_title']}")
                    st.markdown(f"**Quarter**: {development['quarter']}")
                    
                    if development['subject']:
                        st.markdown("**Subject:**")
                        st.write(development['subject'])
                    
                    if development['description']:
                        st.markdown("**Description:**")
                        st.write(development['description'])
                    
                    if development['category']:
                        st.markdown("**Category:**")
                        st.write(development['category'])
        else:
            if search_query.strip():
                st.info("No key developments found matching your search criteria")
            else:
                st.info("No key developments found for the selected time period")
    
    # Footer
    st.markdown("---")
    st.markdown("💡 *Select a company and date to explore key issues, hypotheses, and key developments*")

if __name__ == "__main__":
    main()