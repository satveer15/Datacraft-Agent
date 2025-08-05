import pandas as pd
import streamlit as st
from typing import Optional, Union
import io
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    @staticmethod
    def load_file(uploaded_file) -> Optional[pd.DataFrame]:
        try:
            logger.info(f"Loading file: {uploaded_file.name}")
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:
                logger.error(f"Unsupported file format: {uploaded_file.name}")
                st.error("Unsupported file format. Please upload CSV, Excel, or JSON file.")
                return None
            
            logger.info(f"File loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading file {uploaded_file.name}: {str(e)}")
            st.error(f"Error loading file: {str(e)}")
            return None
    
    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> dict:
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "numeric_columns": list(df.select_dtypes(include=['number']).columns),
            "categorical_columns": list(df.select_dtypes(include=['object']).columns),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2  # in MB
        }
        return summary
    
    @staticmethod
    def get_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
        return df.describe(include='all')