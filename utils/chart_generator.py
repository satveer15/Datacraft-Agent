import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any
import re
import logging

logger = logging.getLogger(__name__)

class ChartGenerator:
    
    @staticmethod
    def detect_chart_request(query: str) -> bool:
        """Check if the query is asking for a chart/plot/visualization"""
        chart_keywords = [
            'plot', 'chart', 'graph', 'visualize', 'show me', 'display',
            'histogram', 'bar chart', 'scatter', 'line chart', 'pie chart',
            'distribution', 'trend', 'correlation', 'heatmap', 'boxplot'
        ]
        query_lower = query.lower()
        result = any(keyword in query_lower for keyword in chart_keywords)
        logger.debug(f"Chart detection for '{query}': {result}")
        return result
    
    @staticmethod
    def parse_chart_request(query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse the query to understand what kind of chart to create"""
        query_lower = query.lower()
        columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        chart_type = None
        if 'histogram' in query_lower or 'distribution' in query_lower:
            chart_type = 'histogram'
        elif 'bar' in query_lower or 'count' in query_lower:
            chart_type = 'bar'
        elif 'scatter' in query_lower or 'correlation' in query_lower:
            chart_type = 'scatter'
        elif 'line' in query_lower or 'trend' in query_lower or 'time' in query_lower:
            chart_type = 'line'
        elif 'pie' in query_lower:
            chart_type = 'pie'
        elif 'box' in query_lower or 'boxplot' in query_lower:
            chart_type = 'box'
        elif 'heatmap' in query_lower:
            chart_type = 'heatmap'
        else:
            chart_type = 'auto'
        mentioned_cols = []
        query_words = re.findall(r'\b\w+\b', query_lower)
        
        for col in columns:
            col_lower = col.lower()
            col_words = re.findall(r'\b\w+\b', col_lower)
            
            if col_lower in query_lower:
                mentioned_cols.append(col)
            elif any(word in col_lower for word in query_words if len(word) > 2):
                mentioned_cols.append(col)
            elif any(word in query_lower for word in col_words if len(word) > 2):
                mentioned_cols.append(col)
        
        mentioned_cols = list(dict.fromkeys(mentioned_cols))
        
        return {
            'chart_type': chart_type,
            'columns': mentioned_cols,
            'numeric_cols': numeric_cols,
            'cat_cols': cat_cols,
            'query': query
        }
    
    @staticmethod
    def create_chart(df: pd.DataFrame, chart_info: Dict[str, Any]) -> Optional[go.Figure]:
        """Create a chart based on the parsed information"""
        try:
            logger.info(f"Creating chart of type: {chart_info['chart_type']}")
            if len(df) > 50000:
                df = df.sample(n=10000, random_state=42)
                st.info(f"📊 Sampling 10,000 rows from large dataset for visualization")
                logger.info("Dataset sampled for performance")
            chart_type = chart_info['chart_type']
            mentioned_cols = chart_info['columns']
            numeric_cols = chart_info['numeric_cols']
            cat_cols = chart_info['cat_cols']
            
            if chart_type == 'auto':
                if len(mentioned_cols) == 1:
                    if mentioned_cols[0] in numeric_cols:
                        chart_type = 'histogram'
                    else:
                        chart_type = 'bar'
                elif len(mentioned_cols) == 2:
                    if all(col in numeric_cols for col in mentioned_cols):
                        chart_type = 'scatter'
                    else:
                        chart_type = 'bar'
                elif len(numeric_cols) > 0:
                    chart_type = 'histogram'
                else:
                    chart_type = 'bar'
                logger.info(f"Auto-detected chart type: {chart_type}")
            if chart_type == 'histogram':
                col = mentioned_cols[0] if mentioned_cols else numeric_cols[0] if numeric_cols else df.columns[0]
                fig = px.histogram(df, x=col, title=f'Distribution of {col}')
                
            elif chart_type == 'bar':
                if len(mentioned_cols) >= 2:
                    x_col, y_col = mentioned_cols[0], mentioned_cols[1]
                    # Aggregate if needed
                    if x_col in cat_cols and y_col in numeric_cols:
                        agg_df = df.groupby(x_col)[y_col].sum().reset_index()
                        fig = px.bar(agg_df, x=x_col, y=y_col, title=f'{y_col} by {x_col}')
                    else:
                        fig = px.bar(df, x=x_col, y=y_col, title=f'{y_col} vs {x_col}')
                elif mentioned_cols and mentioned_cols[0] in cat_cols:
                    col = mentioned_cols[0]
                    value_counts = df[col].value_counts().head(20)
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                                title=f'Top {col} values', labels={'x': col, 'y': 'Count'})
                else:
                    col = cat_cols[0] if cat_cols else df.columns[0]
                    value_counts = df[col].value_counts().head(20)
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                                title=f'Top {col} values', labels={'x': col, 'y': 'Count'})
            
            elif chart_type == 'scatter':
                if len(mentioned_cols) >= 2:
                    x_col, y_col = mentioned_cols[0], mentioned_cols[1]
                elif len(numeric_cols) >= 2:
                    x_col, y_col = numeric_cols[0], numeric_cols[1]
                else:
                    return None
                
                # Add color dimension if available
                color_col = None
                if len(cat_cols) > 0:
                    color_col = cat_cols[0]
                
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                               title=f'{y_col} vs {x_col}')
            
            elif chart_type == 'line':
                # Look for date/time column
                date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                if date_cols:
                    x_col = date_cols[0]
                elif mentioned_cols:
                    x_col = mentioned_cols[0]
                else:
                    x_col = df.columns[0]
                
                y_col = mentioned_cols[1] if len(mentioned_cols) > 1 else numeric_cols[0] if numeric_cols else df.columns[1]
                
                # Sort by x column for line chart
                df_sorted = df.sort_values(by=x_col)
                fig = px.line(df_sorted, x=x_col, y=y_col, title=f'{y_col} over {x_col}')
            
            elif chart_type == 'pie':
                col = mentioned_cols[0] if mentioned_cols else cat_cols[0] if cat_cols else df.columns[0]
                value_counts = df[col].value_counts().head(10)
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f'Distribution of {col}')
            
            elif chart_type == 'box':
                if mentioned_cols:
                    if mentioned_cols[0] in numeric_cols:
                        fig = px.box(df, y=mentioned_cols[0], title=f'Box plot of {mentioned_cols[0]}')
                    elif len(mentioned_cols) >= 2:
                        fig = px.box(df, x=mentioned_cols[0], y=mentioned_cols[1],
                                   title=f'{mentioned_cols[1]} by {mentioned_cols[0]}')
                    else:
                        fig = px.box(df, y=numeric_cols[0] if numeric_cols else df.columns[0],
                                   title='Box plot')
                else:
                    fig = px.box(df, y=numeric_cols[0] if numeric_cols else df.columns[0],
                               title='Box plot')
            
            elif chart_type == 'heatmap':
                # Create correlation heatmap for numeric columns
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, text_auto=True,
                                  title='Correlation Heatmap',
                                  color_continuous_scale='RdBu_r')
                else:
                    return None
            
            else:
                return None
            
            if fig:
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                logger.info(f"Chart created successfully: {chart_type}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating chart: {str(e)}")
            st.error(f"Error creating chart: {str(e)[:200]}")
            return None
    
    @staticmethod
    def generate_chart_from_query(df: pd.DataFrame, query: str) -> Optional[go.Figure]:
        """Main method to generate a chart from a natural language query"""
        if not ChartGenerator.detect_chart_request(query):
            logger.debug(f"No chart request detected for query: {query}")
            return None
        
        logger.info(f"Processing chart request: {query}")
        chart_info = ChartGenerator.parse_chart_request(query, df)
        return ChartGenerator.create_chart(df, chart_info)