#!/usr/bin/env python3
"""
Data loader utility for downloading and caching large data files from GitHub Releases.
"""

import os
import pickle
import requests
import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import hashlib
import json

# GitHub Release URLs for data files
GITHUB_RELEASE_BASE = "https://github.com/t0r0id/intrinsica-apps/releases/download"
DATA_RELEASE_TAG = "v1.0-data"  # Update this when you create the release

# File configurations with expected sizes and checksums (update after upload)
DATA_FILES = {
    "all_documents_with_embeddings.pkl": {
        "url": f"{GITHUB_RELEASE_BASE}/{DATA_RELEASE_TAG}/all_documents_with_embeddings.pkl",
        "expected_size": 187891050,  # 179MB in bytes
        "description": "Document embeddings and metadata"
    },
    "bm25_index.pkl": {
        "url": f"{GITHUB_RELEASE_BASE}/{DATA_RELEASE_TAG}/bm25_index.pkl",
        "expected_size": 31549069,  # 30MB in bytes
        "description": "BM25 search index"
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

def verify_file(file_path: Path, expected_size: Optional[int] = None) -> bool:
    """
    Verify that a file exists and has the expected size.
    
    Args:
        file_path: Path to the file
        expected_size: Expected file size in bytes (optional)
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    if not file_path.exists():
        return False
    
    if expected_size is not None:
        actual_size = file_path.stat().st_size
        # Allow 5% tolerance for size differences
        if abs(actual_size - expected_size) > expected_size * 0.05:
            return False
    
    # Try to load pickle file to verify it's not corrupted
    try:
        with open(file_path, 'rb') as f:
            # Just try to load the pickle header to verify it's valid
            pickle.load(f)
        return True
    except:
        return False

@st.cache_data(show_spinner=False)
def load_data_files() -> Tuple[Dict[str, Any], Any]:
    """
    Load data files, downloading from GitHub Releases if necessary.
    
    Returns:
        Tuple containing:
        - Combined documents data (dict)
        - BM25 index object
    """
    data_dir = ensure_data_dir()
    
    # Check and download each required file
    for filename, config in DATA_FILES.items():
        file_path = data_dir / filename
        
        # Check if file exists and is valid
        if not verify_file(file_path, config.get("expected_size")):
            st.info(f"📥 {config['description']} not found locally. Downloading from GitHub Releases...")
            
            # Download the file
            if not download_file(config["url"], file_path, config["description"]):
                st.error(f"Failed to download {filename}. Please check your internet connection and try again.")
                st.stop()
            
            # Verify downloaded file
            if not verify_file(file_path, config.get("expected_size")):
                st.error(f"Downloaded {filename} appears to be corrupted. Please try again.")
                # Remove corrupted file
                file_path.unlink(missing_ok=True)
                st.stop()
            
            st.success(f"✅ Successfully downloaded {config['description']}")
    
    # Load the data files
    combined_path = data_dir / "all_documents_with_embeddings.pkl"
    bm25_path = data_dir / "bm25_index.pkl"
    
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

def check_data_availability() -> Dict[str, bool]:
    """
    Check which data files are available locally.
    
    Returns:
        Dict mapping filename to availability status
    """
    data_dir = get_data_dir()
    availability = {}
    
    for filename, config in DATA_FILES.items():
        file_path = data_dir / filename
        availability[filename] = verify_file(file_path, config.get("expected_size"))
    
    return availability

def get_total_download_size() -> float:
    """
    Get total size of files that need to be downloaded.
    
    Returns:
        Total size in MB
    """
    data_dir = get_data_dir()
    total_size = 0
    
    for filename, config in DATA_FILES.items():
        file_path = data_dir / filename
        if not verify_file(file_path, config.get("expected_size")):
            total_size += config.get("expected_size", 0)
    
    return total_size / 1024 / 1024  # Convert to MB

def clear_cached_data():
    """Clear cached data files (useful for testing or updates)."""
    data_dir = get_data_dir()
    
    for filename in DATA_FILES.keys():
        file_path = data_dir / filename
        if file_path.exists():
            file_path.unlink()
            st.info(f"Removed {filename}")
    
    # Clear Streamlit cache
    st.cache_data.clear()
    st.success("✅ Cached data cleared successfully")