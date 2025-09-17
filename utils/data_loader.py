#!/usr/bin/env python3
"""
Data loader utility for downloading and caching large data files from GitHub Releases.
"""

import pickle
import requests
import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# GitHub Release URLs for data files
GITHUB_RELEASE_BASE = "https://github.com/t0r0id/intrinsica-apps/releases/download"
DATA_RELEASE_TAG = "v1.0-data"  # Update this when you create the release

# File configurations with expected sizes and checksums (update after upload)
DATA_FILES = {
    "v1":{
        "all_documents_with_embeddings_v1.pkl": {
            "url": f"{GITHUB_RELEASE_BASE}/{DATA_RELEASE_TAG}/all_documents_with_embeddings_v1.pkl",

            "description": "Document embeddings and metadata"
        },
        "bm25_index_v1.pkl": {
            "url": f"{GITHUB_RELEASE_BASE}/{DATA_RELEASE_TAG}/bm25_index_v1.pkl",
            "description": "BM25 search index"
        }
    },
    "v2":{
        "all_documents_with_embeddings_v2.pkl": {
            "url": f"{GITHUB_RELEASE_BASE}/{DATA_RELEASE_TAG}/all_documents_with_embeddings_v2.pkl",
            "description": "Document embeddings and metadata"
        },
        "bm25_index_v2.pkl": {
            "url": f"{GITHUB_RELEASE_BASE}/{DATA_RELEASE_TAG}/bm25_index_v2.pkl",
            "description": "BM25 search index"
        }
    }
}

def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent.parent / "data"

def ensure_data_dir() -> Path:
    """Ensure data directory exists."""
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def download_file(url: str, destination: Path, description: str = "file") -> bool:
    """
    Download a file from URL with progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save the file
        description: Description for progress bar
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Start download with stream=True for progress tracking
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Create progress bar
        progress_text = f"Downloading {description}..."
        progress_bar = st.progress(0, text=progress_text)
        
        # Download in chunks
        chunk_size = 8192
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Update progress
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(
                            progress,
                            text=f"{progress_text} ({downloaded / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MB)"
                        )
        
        progress_bar.empty()
        return True
        
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download {description}: {str(e)}")
        return False
    except Exception as e:
        st.error(f"Unexpected error downloading {description}: {str(e)}")
        return False

def verify_file(file_path: Path) -> bool:
    """
    Verify that a file exists and has the expected size.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    if not file_path.exists():
        return False
    return True

@st.cache_data(show_spinner=False)
def load_data_files(version: Optional[str] = "v1") -> Tuple[Dict[str, Any], Any]:
    """
    Load data files, downloading from GitHub Releases if necessary.
    
    Returns:
        Tuple containing:
        - Combined documents data (dict)
        - BM25 index object
    """
    data_dir = ensure_data_dir()
    
    # Check and download each required file
    for filename, config in DATA_FILES[version].items():
        file_path = data_dir / filename
        
        # Check if file exists and is valid
        if not verify_file(file_path):
            # Download the file silently
            if not download_file(config["url"], file_path, config["description"]):
                st.error(f"Failed to download {filename}. Please check your internet connection and try again.")
                st.stop()
            
            # Verify downloaded file
            if not verify_file(file_path):
                st.error(f"Downloaded {filename} appears to be corrupted. Please try again.")
                # Remove corrupted file
                file_path.unlink(missing_ok=True)
                st.stop()
            
            # Download completed successfully - no status message needed
    
    # Load the data files
    combined_path = data_dir / f"all_documents_with_embeddings_{version}.pkl"
    bm25_path = data_dir / f"bm25_index_{version}.pkl"
    
    try:
        with open(combined_path, 'rb') as f:
            combined_data = pickle.load(f)
        
        with open(bm25_path, 'rb') as f:
            bm25_index = pickle.load(f)
        
        return combined_data, bm25_index
        
    except Exception as e:
        st.error(f"Error loading data files: {str(e)}")
        st.error("The data files may be corrupted. Try deleting them and refreshing the page to re-download.")
        st.stop()




