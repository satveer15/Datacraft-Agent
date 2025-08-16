import streamlit as st
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def init_chat_session():
    """Initialize chat session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        logger.debug("Initialized chat messages")
    if 'chat_context' not in st.session_state:
        st.session_state.chat_context = []
        logger.debug("Initialized chat context")
    if 'df_context' not in st.session_state:
        st.session_state.df_context = None
        logger.debug("Initialized dataframe context")

def add_message(role, content, timestamp=None):
    """Add a message to the chat history"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M")
    
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": timestamp
    })
    
    st.session_state.chat_context.append({
        "role": role,
        "content": content
    })
    if len(st.session_state.chat_context) > 10:
        st.session_state.chat_context = st.session_state.chat_context[-10:]
    
    logger.debug(f"Added {role} message to chat history")

def display_chat_message(message):
    """Display a single chat message with proper styling"""
    role = message["role"]
    content = message["content"]
    timestamp = message.get("timestamp", "")
    
    if role == "user":
        with st.container():
            col1, col2 = st.columns([10, 1])
            with col1:
                st.markdown(
                    f"""
                    <div style="background-color: #e3f2fd; padding: 10px 15px; border-radius: 15px; margin: 5px 0; text-align: right;">
                        <div style="color: #1976d2; font-size: 12px; margin-bottom: 5px;">You • {timestamp}</div>
                        <div style="color: #333;">{content}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        with st.container():
            col1, col2 = st.columns([1, 10])
            with col2:
                st.markdown(
                    f"""
                    <div style="background-color: #f5f5f5; padding: 10px 15px; border-radius: 15px; margin: 5px 0;">
                        <div style="color: #666; font-size: 12px; margin-bottom: 5px;">AI Assistant • {timestamp}</div>
                        <div style="color: #333;">{content}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

def create_context_prompt(query, df_info=None):
    """Create a context-aware prompt for the AI"""
    context_parts = []
    
    # Add conversation history
    if st.session_state.chat_context:
        context_parts.append("Previous conversation:")
        for msg in st.session_state.chat_context[-5:]:  # Last 5 messages for context
            context_parts.append(f"{msg['role']}: {msg['content']}")
    
    # Add data context if available
    if df_info:
        context_parts.append(f"\nData context: {df_info}")
    
    # Add current query
    context_parts.append(f"\nCurrent question: {query}")
    
    return "\n".join(context_parts)

def get_suggested_questions(df: pd.DataFrame):
    """Generate suggested questions based on the data"""
    logger.debug("Generating suggested questions based on data structure")
    suggestions = []
    
    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    suggestions.append("What are the main patterns in this data?")
    suggestions.append("Summarize the key statistics")
    if numeric_cols:
        col = numeric_cols[0]
        suggestions.append(f"Plot the distribution of {col}")
        suggestions.append(f"Show me a histogram of {col}")
        if len(numeric_cols) > 1:
            suggestions.append(f"Create a scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}")
    
    if cat_cols:
        col = cat_cols[0]
        suggestions.append(f"Show me a bar chart of {col} counts")
        suggestions.append(f"Create a pie chart of {col} distribution")
    
    if numeric_cols and cat_cols:
        suggestions.append(f"Plot {numeric_cols[0]} by {cat_cols[0]}")
        suggestions.append(f"Show me a box plot of {numeric_cols[0]} grouped by {cat_cols[0]}")
    
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if date_cols and numeric_cols:
        suggestions.append(f"Plot {numeric_cols[0]} over time")
    
    logger.debug(f"Generated {len(suggestions)} suggested questions")
    return suggestions[:8]