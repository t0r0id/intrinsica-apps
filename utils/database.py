"""
Database utilities for Intrinsica Apps
"""

import streamlit as st
import os
from sqlalchemy import create_engine, Column, Integer, TEXT, Date, Boolean, ForeignKey, TIMESTAMP, func, ARRAY
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Models
Base = declarative_base()

class BaseWithTimeStamp(Base):
    __abstract__ = True
    id = Column(Integer, primary_key=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    last_updated_at = Column(TIMESTAMP, server_default=func.now(), server_onupdate=func.now())

class Company(BaseWithTimeStamp):
    __tablename__ = "companies"
    name = Column(TEXT, nullable=False)
    ticker = Column(TEXT, nullable=False, unique=True)
    exchange = Column(TEXT, nullable=False)
    is_active = Column(Boolean, default=True)

class Transcript(BaseWithTimeStamp):
    __tablename__ = "transcripts"
    company_id = Column(Integer, ForeignKey("companies.id", ondelete="CASCADE"))
    title = Column(TEXT, nullable=False)
    published_date = Column(Date, nullable=False)
    quarter = Column(TEXT, nullable=False)

class ConferenceInsightsReport(BaseWithTimeStamp):
    __tablename__ = "conference_insights_reports"
    transcript_id = Column(Integer, ForeignKey("transcripts.id", ondelete="CASCADE"))
    is_active = Column(Boolean, nullable=False)

class TranscriptKeyIssue(BaseWithTimeStamp):
    __tablename__ = "transcript_key_issues"
    conference_insights_report_id = Column(Integer, ForeignKey("conference_insights_reports.id", ondelete="CASCADE"))
    title = Column(TEXT)
    analyst_concern = Column(TEXT)
    management_response = Column(TEXT)
    counterpoints = Column(ARRAY(TEXT))
    followup_questions = Column(ARRAY(TEXT))

class HypothesisState(BaseWithTimeStamp):
    __tablename__ = "hypotheses_states"
    source_key_development_id = Column(Integer, ForeignKey("transcript_key_developments.id", ondelete="CASCADE"))

class Hypothesis(BaseWithTimeStamp):
    __tablename__ = "hypotheses"
    target_company_id = Column(Integer, ForeignKey("companies.id", ondelete="CASCADE"), nullable=False)
    hypothesis_state_id = Column(Integer, ForeignKey("hypotheses_states.id", ondelete="CASCADE"), nullable=False)
    title = Column(TEXT)
    reasoning = Column(TEXT)

class TranscriptKeyDevelopment(BaseWithTimeStamp):
    __tablename__ = "transcript_key_developments"
    conference_insights_report_id = Column(Integer, ForeignKey("conference_insights_reports.id", ondelete="CASCADE"))

class PublishedRadarReport(BaseWithTimeStamp):
    __tablename__ = "published_radar_reports"
    target_company_id = Column(Integer, ForeignKey("companies.id", ondelete="CASCADE"), nullable=False)
    title = Column(TEXT)
    published_date = Column(Date, nullable=False)

@st.cache_resource
def get_database_connection():
    """Create database connection using environment variables"""
    try:
        # Get database credentials from environment
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD") 
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")
        db_name = os.getenv("DB_NAME")
        
        if not all([db_user, db_password, db_host, db_port, db_name]):
            st.error("Missing database environment variables. Please check your .env file.")
            st.stop()
        
        database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        engine = create_engine(
            database_url,
            pool_recycle=300,
            pool_pre_ping=True,
            pool_size=3,
            max_overflow=0,
        )
        
        SessionLocal = sessionmaker(bind=engine)
        return SessionLocal
        
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        st.stop()