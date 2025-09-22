#!/usr/bin/env python3
"""
Streamlit UI for Business Unit Competitor Identification.
Uses hybrid search (BM25 + Semantic Similarity) to find competitors.
"""

import streamlit as st
import pickle
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import norm
from utils.auth import check_authentication
from utils.data_loader import load_data_files

# Check authentication
if not check_authentication():
    st.error("Please login from the main page to access this application.")
    st.stop()

# Page configuration for this specific page
st.set_page_config(
    page_title="Competitor Identification",
    page_icon="🎯",
    layout="wide"
)

@st.cache_data
def load_indices(data_version: str = "v1"):
    """Load pre-computed indices and data from GitHub Releases if needed."""
    # Load data files (will download from GitHub Releases if not present)
    if data_version == "v2":
        combined_data, bm25_index = load_data_files("v2")
    else:
        combined_data, bm25_index = load_data_files()
    
    # Extract documents, embeddings, and tokenized texts
    all_documents = []
    embeddings_list = []
    tokenized_texts = []
    
    for entry in combined_data:
        # Extract document info (excluding embedding and tokenized_text)
        doc = {k: v for k, v in entry.items() 
               if k not in ['embedding', 'tokenized_text']}
        
        # Set default importance weight for 10K documents if missing
        if doc.get('document_type') == '10k' and doc.get('importance_weight') is None:
            doc['importance_weight'] = 100
            
        all_documents.append(doc)
        
        # Extract embedding
        if entry.get('embedding'):
            embeddings_list.append(entry['embedding'])
        else:
            st.error(f"Missing embedding for document: {doc.get('name', 'Unknown')}")
            st.stop()
        
        # Extract tokenized text
        if entry.get('tokenized_text'):
            tokenized_texts.append(entry['tokenized_text'])
        else:
            st.error(f"Missing tokenized text for document: {doc.get('name', 'Unknown')}")
            st.stop()
    
    # Convert embeddings to numpy array
    embeddings = np.array(embeddings_list)
    
    # Create processed_data dictionary
    processed_data = {
        'all_documents': all_documents,
        'business_units': [doc for doc in all_documents if doc.get('document_type') == 'business_unit'],
        '10k_documents': [doc for doc in all_documents if doc.get('document_type') == '10k'],
        'tokenized_texts': tokenized_texts
    }
    
    # Validate data consistency
    if len(embeddings) != len(all_documents):
        st.error(f"Data inconsistency detected! Embeddings: {len(embeddings)}, Documents: {len(all_documents)}")
        st.error("Data files may be corrupted. Please clear cache and refresh.")
        st.stop()
    
    if len(tokenized_texts) != len(all_documents):
        st.error(f"Data inconsistency detected! Tokenized texts: {len(tokenized_texts)}, Documents: {len(all_documents)}")
        st.error("Data files may be corrupted. Please clear cache and refresh.")
        st.stop()
    
    return embeddings, bm25_index, processed_data

def compute_semantic_similarity(query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and document embeddings."""
    # Normalize embeddings
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    
    # Compute cosine similarity
    similarities = np.dot(doc_norms, query_norm)
    
    return similarities

def normalize_scores_min_max(scores: np.ndarray) -> np.ndarray:
    """Normalize scores using standard normal distribution (z-score normalization)."""
    if len(scores) == 0:
        return scores
    
    min, max = scores.min(), scores.max()
    normalized = (scores-min)/(max-min)
    return normalized

def normalize_scores_zdist(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    
    # Calculate mean and standard deviation
    mu = np.mean(scores)
    sigma = np.std(scores, ddof=1)  # Use sample standard deviation
    
    if sigma == 0:
        # All scores are the same
        return np.zeros_like(scores)
    
    # Z-score normalization
    z_scores = (scores - mu) / sigma
     # Convert z-scores to [0, 1] range using sigmoid function
    # This maps z-scores to a 0-1 range while preserving relative distances
    normalized = norm.cdf(z_scores)
    return normalized

def get_similarity_scores(
    source_indices: list,
    target_indices: list,
    embeddings: np.ndarray,
    bm25_index,
    tokenized_texts: list,
    bm25_weight: float = 0.7,
    semantic_weight: float = 0.3,

) -> dict:
    """
    Get similarity scores between source documents and target documents.
    
    Args:
        source_indices: List of indices for source documents
        target_indices: List of indices for target documents
        embeddings: Array of embeddings for all documents
        bm25_index: Pre-computed BM25 index
        tokenized_texts: List of tokenized texts for all documents
        bm25_weight: Weight for BM25 scores
        semantic_weight: Weight for semantic scores
    Returns:
        dict: Dictionary containing:
            - 'combined': List[List[float]] - Combined similarity scores
            - 'bm25_normalized': List[List[float]] - Normalized BM25 scores
            - 'semantic_normalized': List[List[float]] - Normalized semantic scores
            Where scores[i][j] is the similarity between source_indices[i] and target_indices[j]
    """
    combined_scores = []
    bm25_normalized_scores = []
    semantic_normalized_scores = []
    
    for src_idx in source_indices:
        # Get source embedding and tokens
        src_embedding = embeddings[src_idx]
        src_tokens = tokenized_texts[src_idx]
        
        # Compute BM25 scores for this source unit against all units
        all_bm25_scores = np.array(bm25_index.get_scores(src_tokens))
        
        # Compute semantic similarity scores against all units
        all_semantic_scores = compute_semantic_similarity(src_embedding, embeddings)
        
        # Get scores only for target units
        target_bm25 = all_bm25_scores[target_indices]
        target_semantic = all_semantic_scores[target_indices]

        
        # Normalize scores if we have multiple targets
        if len(target_indices) > 1:
            target_bm25_norm = normalize_scores_zdist(target_bm25)
            target_semantic_norm = target_semantic
        else:
            # Single target - use min-max normalization
            target_bm25_norm = target_bm25 / (np.max(target_bm25) + 1e-10)
            target_semantic_norm = target_semantic / (np.max(target_semantic) + 1e-10)
        
        # Combine scores
        combined = (bm25_weight * target_bm25_norm) + (semantic_weight * target_semantic_norm)
        
        combined_scores.append(combined.tolist())
        bm25_normalized_scores.append(target_bm25_norm.tolist())
        semantic_normalized_scores.append(target_semantic_norm.tolist())
    
    return {
        'combined': combined_scores,
        'bm25_normalized': bm25_normalized_scores,
        'semantic_normalized': semantic_normalized_scores
    }

def find_competitors(
    query_unit_idx: int,
    embeddings: np.ndarray,
    bm25_index,
    tokenized_texts: list,
    documents: list,
    bm25_weight: float = 0.7,
    semantic_weight: float = 0.3
) -> pd.DataFrame:
    """Find competitors using hybrid search."""

    # Get target units (all units except those from source company)
    target_indices = [i for i, doc in enumerate(documents) if doc['ticker'] != documents[query_unit_idx]['ticker']]

    # Get similarity scores using the new method
    # Source: [query_unit_idx], Targets: all units except source company
    similarity_results = get_similarity_scores(
        source_indices=[query_unit_idx],
        target_indices=target_indices,
        embeddings=embeddings,
        bm25_index=bm25_index,
        tokenized_texts=tokenized_texts,
        bm25_weight=bm25_weight,
        semantic_weight=semantic_weight,
    )

    # Extract scores for the single source unit (first and only row)
    final_scores = similarity_results['combined'][0]
    bm25_norm = similarity_results['bm25_normalized'][0]
    semantic_norm = similarity_results['semantic_normalized'][0]

    # Create results dataframe with all units
    all_results = []
    for idx, target_idx in enumerate(target_indices):
        doc = documents[target_idx]
        all_results.append({
            'display_name': doc['display_name'],
            'ticker': doc['ticker'],
            'name': doc['name'],
            'bm25_score': bm25_norm[idx],
            'semantic_score': semantic_norm[idx],
            'final_score': final_scores[idx],
            'details': doc['details']  # Store full details
        })

    # Convert to DataFrame for easier manipulation
    all_results_df = pd.DataFrame(all_results)

    # Group by company ticker and keep only the best scoring unit per company
    best_per_company = []
    for ticker, group in all_results_df.groupby('ticker'):
        # Get the row with highest final_score for this company
        best_unit = group.loc[group['final_score'].idxmax()]
        best_per_company.append(best_unit)

    # Create final results dataframe
    results_df = pd.DataFrame(best_per_company)
    results_df = results_df.sort_values('final_score', ascending=False)

    return results_df

def find_company_competitors(
    source_company: str,
    embeddings: np.ndarray,
    bm25_index,
    tokenized_texts: list,
    documents: list,
    document_type: str = 'business_unit',
    bm25_weight: float = 0.7,
    semantic_weight: float = 0.3
) -> pd.DataFrame:
    """Find company-level competitors by computing average best match scores."""
    # Filter documents by type
    filtered_documents = [doc for doc in documents if doc.get('document_type') == document_type]
    doc_to_original_idx = {i: documents.index(doc) for i, doc in enumerate(filtered_documents)}
    
    # Get all documents for source company
    source_indices = [doc_to_original_idx[i] for i, doc in enumerate(filtered_documents) if doc['ticker'] == source_company]
    
    if not source_indices:
        return pd.DataFrame()
    
    # Get all target documents (excluding source company)
    target_indices = [doc_to_original_idx[i] for i, doc in enumerate(filtered_documents)]
    
    if not target_indices:
        return pd.DataFrame()
    
    # Get similarity scores between all source units and all other units at once
    similarity_results = get_similarity_scores(
        source_indices=source_indices,
        target_indices=target_indices,
        embeddings=embeddings,
        bm25_index=bm25_index,
        tokenized_texts=tokenized_texts,
        bm25_weight=bm25_weight,
        semantic_weight=semantic_weight,
    )

    # Build company display name map and company-to-indices map
    company_display_map = {}
    company_indices_map = {}
    
    for idx in target_indices:
        doc = documents[idx]
        ticker = doc['ticker']
        
        if ticker not in company_display_map:
            company_display_map[ticker] = doc.get('company_display', ticker)
            company_indices_map[ticker] = []
        
        # Map each target index to its position in target_indices list
        company_indices_map[ticker].append(target_indices.index(idx))
    
    # For each source unit, find the maximum score for each target company
    company_scores = []
    
    for target_company, target_positions in company_indices_map.items():
        best_matches = []
        source_weights = []
        matched_pairs = []
        
        # For each source unit
        for i, src_idx in enumerate(source_indices):
            # Get scores for this source unit against all target units
            scores_for_source = similarity_results['combined'][i]
            
            # Get scores only for this target company's units
            company_scores_subset = [scores_for_source[pos] for pos in target_positions]
            
            if company_scores_subset:
                # Find the best match among this company's units
                best_score = max(company_scores_subset)
                best_idx_in_subset = company_scores_subset.index(best_score)
                best_target_position = target_positions[best_idx_in_subset]
                best_target_idx = target_indices[best_target_position]
                
                best_matches.append(best_score)
                source_weights.append(documents[src_idx].get('importance_weight', 100))
                matched_pairs.append({
                    'source': documents[src_idx]['name'],
                    'target': documents[best_target_idx]['name'],
                    'score': best_score
                })
        
        # Calculate weighted average score across all source units for this company
        if best_matches:
            weights = np.array(source_weights)
            scores = np.array(best_matches)
            avg_score = np.sum(scores * weights) / np.sum(weights)
            
            # Get company business units info
            target_units = [documents[target_indices[pos]]['name'] for pos in target_positions]
            
            company_scores.append({
                'ticker': target_company,
                'display_name': company_display_map[target_company],
                'avg_score': avg_score,
                'num_matches': len(best_matches),
                'business_units': ', '.join(target_units[:3]) + ('...' if len(target_units) > 3 else ''),
                'matched_pairs': matched_pairs
            })
    
    # Create dataframe and normalize final scores
    results_df = pd.DataFrame(company_scores)
    
    if not results_df.empty:
        # Normalize the average scores across all companies
        results_df['final_score'] = results_df['avg_score'].values
        results_df = results_df.sort_values('final_score', ascending=False)
    
    return results_df

def main():
    """Main Streamlit application."""
    st.title("🎯 Business Unit Competitor Identification")
    st.markdown("Find competitors based on business unit descriptions using hybrid search (BM25 + Semantic Similarity)")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Data version selection
        st.subheader("Data Version")
        data_version = st.selectbox(
            "Select Data Version",
            ["v1", "v2"],
            help="Choose between different versions of the data"
        )

        # Weights configuration
        st.subheader("Search Weights")
        bm25_weight = st.slider("BM25 Weight", 0.0, 1.0, 0.7, 0.05)
        semantic_weight = 1.0 - bm25_weight
        st.info(f"Semantic Weight: {semantic_weight:.2f}")

        # Number of results
        top_n = st.number_input("Top N Results", min_value=5, max_value=50, value=10)
        
        # Scoring methodology explanation
        with st.expander("📊 Scoring Methodology"):
            st.markdown("""
            **Score Normalization:**
            - Uses **standard normal distribution** (z-score normalization)
            - Scores are transformed using sigmoid function to [0, 1] range
            - Higher scores indicate stronger similarity
            
            **Hybrid Search:**
            - BM25: Keyword-based matching
            - Semantic: Embedding-based similarity
            - Combined using configurable weights
            """)
    
    try:
        with st.spinner(f"Loading data files (Version: {data_version})..."):
            embeddings, bm25_index, processed_data = load_indices(data_version)
            all_documents = processed_data['all_documents']
            business_units = processed_data['business_units']
            raw_10k_docs = processed_data['10k_documents']
            tokenized_texts = processed_data['tokenized_texts']

        st.success(f"✅ Loaded {len(all_documents)} documents ({len(business_units)} business units, {len(raw_10k_docs)} 10K docs) from {len(set([d['ticker'] for d in all_documents]))} companies (Version: {data_version})")
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error("The index files may be corrupted. Please regenerate them:")
        st.code("cd experiments/graph\npython index_business_units.py --full", language="bash")
        st.stop()
    
    # Company and Business Unit Selection
    col1, col2 = st.columns(2)
    
    with col1:
        # Get unique companies with their display names
        company_map = {}
        for unit in business_units:
            if unit['ticker'] not in company_map:
                # Use company_display if available, otherwise just ticker
                company_map[unit['ticker']] = unit.get('company_display', unit['ticker'])
        
        # Sort by company name
        sorted_companies = sorted(company_map.items(), key=lambda x: x[1])
        company_display_to_ticker = {display: ticker for ticker, display in sorted_companies}
        
        selected_company_display = st.selectbox("Select Company", [display for _, display in sorted_companies])
        selected_company = company_display_to_ticker[selected_company_display]
    
    with col2:
        # Get business units for selected company
        company_units = [unit for unit in business_units if unit['ticker'] == selected_company]
        
        # Add "All Business Units" and "Raw 10K" options
        unit_options = {
            "📊 All Business Units (Company Average)": -1,
            "📄 Raw 10K (Company Average)": -2
        }
        unit_options.update({unit['display_name']: idx for idx, unit in enumerate(business_units) if unit['ticker'] == selected_company})
        
        if len(unit_options) > 1:  # Has units besides "All"
            selected_unit_display = st.selectbox("Select Business Unit", list(unit_options.keys()))
            selected_unit_idx = unit_options[selected_unit_display]
        else:
            st.error("No business units found for selected company")
            st.stop()
    
    # Search button
    if st.button("🔍 Find Competitors", type="primary"):
        with st.spinner("Searching for competitors..."):
            
            if selected_unit_idx == -1:
                # Company-level search using business units
                st.info(f"Finding company-level competitors for {selected_company} based on average similarity across all business units...")
                
                results_df = find_company_competitors(
                    selected_company,
                    embeddings,
                    bm25_index,
                    tokenized_texts,
                    all_documents,
                    document_type='business_unit',
                    bm25_weight=bm25_weight,
                    semantic_weight=semantic_weight
                )

                # Get top N results
                top_results = results_df.head(top_n)
                
                # Display results
                st.header(f"🏆 Top Company Competitors for {selected_company}")
                
                # Create tabs for company view
                tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "📋 Table View", "🔗 Match Details"])
            elif selected_unit_idx == -2:
                # Company-level search using Raw 10K documents
                st.info(f"Finding company-level competitors for {selected_company} based on Raw 10K document similarity...")
                
                results_df = find_company_competitors(
                    selected_company,
                    embeddings,
                    bm25_index,
                    tokenized_texts,
                    all_documents,
                    document_type='10k',
                    bm25_weight=bm25_weight,
                    semantic_weight=semantic_weight
                )
                
                # Get top N results
                top_results = results_df.head(top_n)
                
                # Display results
                st.header(f"🏆 Top Company Competitors for {selected_company}")
                
                # Create tabs for company view
                tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "📋 Table View", "🔗 Match Details"])
            else:
                # Individual unit search
                st.info(f"Finding competitors for specific business unit: {selected_unit_display}")
                
                # Find the actual document index from all_documents
                selected_doc_idx = None
                for i, doc in enumerate(all_documents):
                    if (doc.get('ticker') == selected_company and 
                        doc.get('display_name') == selected_unit_display and
                        doc.get('document_type') == 'business_unit'):
                        selected_doc_idx = i
                        break
                
                if selected_doc_idx is None:
                    st.error("Selected business unit not found in document index.")
                    st.stop()
                
                results_df = find_competitors(
                    selected_doc_idx,
                    embeddings,
                    bm25_index,
                    tokenized_texts,
                    [d for d in all_documents if d['document_type'] == 'business_unit'],
                    bm25_weight=bm25_weight,
                    semantic_weight=semantic_weight
                )
                
                # Get top N results
                top_results = results_df.head(top_n)
                
                # Display results
                st.header("🏆 Top Competitors")
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "📋 Table View", "📝 Detailed View"])
            
            # Tab content rendering (moved outside conditional blocks)
            with tab1:
                # Create bar chart
                fig = go.Figure()
                
                if selected_unit_idx == -1 or selected_unit_idx == -2:
                    # Company-level bar chart
                    fig.add_trace(go.Bar(
                        x=top_results['final_score'].values,
                        y=top_results['display_name'].values,
                        orientation='h',
                        name='Average Score',
                        marker_color='rgb(55, 83, 109)',
                        text=[f"{score:.3f}" for score in top_results['final_score']],
                        textposition='auto',
                    ))
                    
                    chart_title = "Top Competitor Companies (10K)" if selected_unit_idx == -2 else "Top Competitor Companies (Business Units)"
                    fig.update_layout(
                        title=chart_title,
                        xaxis_title="Average Similarity Score",
                        yaxis_title="Company",
                        height=400 + (len(top_results) * 30),
                        showlegend=False,
                        yaxis={'categoryorder': 'total ascending'}
                    )
                else:
                    # Unit-level bar chart
                    fig.add_trace(go.Bar(
                        x=top_results['final_score'].values,
                        y=top_results['display_name'].values,
                        orientation='h',
                        name='Final Score',
                        marker_color='rgb(55, 83, 109)',
                        text=[f"{score:.3f}" for score in top_results['final_score']],
                        textposition='auto',
                    ))
                    
                    fig.update_layout(
                        title="Top Competitor Business Units",
                        xaxis_title="Similarity Score",
                        yaxis_title="Business Unit",
                        height=400 + (len(top_results) * 20),
                        showlegend=False,
                        yaxis={'categoryorder': 'total ascending'}
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
            with tab2:
                # Display table
                if selected_unit_idx == -1 or selected_unit_idx == -2:
                    # Company-level table
                    display_df = top_results[['display_name', 'final_score', 'num_matches', 'business_units']].copy()
                    display_df.columns = ['Company', 'Average Score', '# Matches', 'Sample Business Units']
                    display_df['Average Score'] = display_df['Average Score'].apply(lambda x: f"{x:.3f}")
                else:
                    # Unit-level table
                    display_df = top_results[['display_name', 'final_score', 'bm25_score', 'semantic_score']].copy()
                    display_df.columns = ['Business Unit', 'Final Score', 'BM25 Score', 'Semantic Score']
                    
                    # Format scores
                    for col in ['Final Score', 'BM25 Score', 'Semantic Score']:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
            with tab3:
                if selected_unit_idx == -1 or selected_unit_idx == -2:
                    # Company match details
                    for _, row in top_results.iterrows():
                        with st.expander(f"**{row['display_name']}** - Average Score: {row['final_score']:.3f}"):
                            st.markdown(f"**Matched {row['num_matches']} business unit pairs**")
                            
                            # Show matched pairs
                            if 'matched_pairs' in row and row['matched_pairs']:
                                pairs_df = pd.DataFrame(row['matched_pairs'])
                                pairs_df['score'] = pairs_df['score'].apply(lambda x: f"{x:.3f}")
                                pairs_df.columns = ['Source Business Unit', 'Best Match', 'Score']
                                st.dataframe(pairs_df, use_container_width=True, hide_index=True)
                            
                else:
                    # Unit detailed view
                    for _, row in top_results.iterrows():
                        with st.expander(f"**{row['display_name']}** - Score: {row['final_score']:.3f}"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("BM25 Score", f"{row['bm25_score']:.3f}")
                            with col2:
                                st.metric("Semantic Score", f"{row['semantic_score']:.3f}")
                            with col3:
                                st.metric("Final Score", f"{row['final_score']:.3f}")
                            
                            st.markdown("**Business Unit Description:**")
                            
                            # Create a scrollable container with proper markdown rendering
                            # Using container with height constraint
                            with st.container(height=200, border=True):
                                st.markdown(row['details'])
            
            # Score distribution - only show for unit-level search
            if selected_unit_idx > -1:
                with st.expander("📈 Score Distribution"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Histogram of all scores
                        all_scores = results_df['final_score'].values
                        fig_hist = px.histogram(
                            x=all_scores,
                            nbins=30,
                            title="Distribution of All Similarity Scores",
                            labels={'x': 'Similarity Score', 'y': 'Count'}
                        )
                        fig_hist.add_vline(
                            x=top_results.iloc[-1]['final_score'],
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Top N Cutoff"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        # Score breakdown for top results
                        score_breakdown = pd.DataFrame({
                            'Business Unit': top_results['display_name'].str[:30] + '...',
                            'BM25': top_results['bm25_score'],
                            'Semantic': top_results['semantic_score']
                        })
                        
                        fig_breakdown = px.bar(
                            score_breakdown.melt(id_vars='Business Unit', var_name='Score Type', value_name='Score'),
                            x='Score',
                            y='Business Unit',
                            color='Score Type',
                            orientation='h',
                            title="Score Breakdown (BM25 vs Semantic)",
                            barmode='group'
                        )
                        st.plotly_chart(fig_breakdown, use_container_width=True)
            else:
                # Company-level distribution
                with st.expander("📈 Score Distribution"):
                    # Show distribution of average scores
                    all_scores = results_df['final_score'].values
                    fig_hist = px.histogram(
                        x=all_scores,
                        nbins=20,
                        title="Distribution of Company Average Scores",
                        labels={'x': 'Average Similarity Score', 'y': 'Count'}
                    )
                    if len(top_results) > 0:
                        fig_hist.add_vline(
                            x=top_results.iloc[-1]['final_score'],
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Top N Cutoff"
                        )
                    st.plotly_chart(fig_hist, use_container_width=True)

if __name__ == "__main__":
    main()