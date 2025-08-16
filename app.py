import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import DataLoader
from utils.chat_ui import init_chat_session, add_message, display_chat_message, create_context_prompt, get_suggested_questions
from agents.data_analyst import DataAnalystAgent
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="DataCraft SuperTool",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ DataCraft SuperTool")
st.markdown("Craft brilliant insights from any dataset with AI precision!")

with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("API Keys")
    
    # Get environment keys (never show these)
    env_openai_key = os.getenv("OPENAI_API_KEY", "")
    env_gemini_key = os.getenv("GOOGLE_API_KEY", "")
    
    # Show which keys are configured via environment
    env_configured = []
    if env_openai_key and env_openai_key not in ["your_working_api_key_here", ""]:
        env_configured.append("OpenAI")
    if env_gemini_key and env_gemini_key not in ["your_gemini_api_key_here", ""]:
        env_configured.append("Gemini")
    
    if env_configured:
        st.success(f"🔒 Environment keys configured: {', '.join(env_configured)}")
        logger.info(f"Environment API keys detected: {', '.join(env_configured)}")
    
    # User input for additional/override keys (never show environment values)
    st.markdown("##### Override or Add Additional Keys:")
    user_openai_key = st.text_input(
        "OpenAI API Key (optional override)", 
        type="password", 
        help="Leave empty to use environment key if available",
        placeholder="sk-..." if not env_openai_key else "Using environment key (leave empty)"
    )
    user_gemini_key = st.text_input(
        "Google Gemini API Key (optional override)", 
        type="password", 
        help="Leave empty to use environment key if available",
        placeholder="AIza..." if not env_gemini_key else "Using environment key (leave empty)"
    )
    
    # Determine final keys to use (user input overrides environment)
    final_openai_key = user_openai_key if user_openai_key else env_openai_key
    final_gemini_key = user_gemini_key if user_gemini_key else env_gemini_key
    
    # Show status
    user_configured = []
    if user_openai_key:
        user_configured.append("OpenAI (override)")
    if user_gemini_key:
        user_configured.append("Gemini (override)")
    
    if user_configured:
        st.info(f"🔑 User override keys: {', '.join(user_configured)}")
        logger.info(f"User provided API key overrides: {', '.join(user_configured)}")
    
    # Final validation
    if not final_openai_key and not final_gemini_key:
        st.warning("⚠️ No API keys available")
        st.info("💡 Add API keys via:")
        st.info("• Environment variables (.env file)")
        st.info("• Input fields above")
        logger.info("No API keys available (environment or user input)")
    else:
        total_configured = []
        if final_openai_key:
            total_configured.append("OpenAI")
        if final_gemini_key:
            total_configured.append("Gemini")
        
        if not user_configured and env_configured:
            # Only environment keys, no user overrides
            pass  # Already shown above
        elif user_configured and not env_configured:
            # Only user keys, no environment
            st.info(f"📝 Active keys: {', '.join(total_configured)}")
        
    # Use final keys for the rest of the application
    openai_key = final_openai_key
    gemini_key = final_gemini_key
    
    st.markdown("---")
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Upload a CSV, Excel, or JSON file"
    )
    
    if uploaded_file:
        st.success(f"Loaded: {uploaded_file.name}")
        logger.info(f"File uploaded: {uploaded_file.name}, size: {uploaded_file.size} bytes")
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if uploaded_file:
    df = DataLoader.load_file(uploaded_file)
    
    if df is not None:
        st.session_state.df = df
        logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        if not st.session_state.agent or openai_key or gemini_key:
            st.session_state.agent = DataAnalystAgent(openai_key, gemini_key)
            logger.info("DataAnalyst agent initialized")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "💬 AI Chat", "📈 Visualizations", "🔍 Insights"])
        
        with tab1:
            st.header("Data Preview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Numeric Columns", len(df.select_dtypes(include=['number']).columns))
            with col4:
                st.metric("Text Columns", len(df.select_dtypes(include=['object']).columns))
            
            st.subheader("Data Sample")
            st.dataframe(df.head(100), use_container_width=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Column Information")
                info_df = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null Count': df.count().values,
                    'Null Count': df.isnull().sum().values
                })
                st.dataframe(info_df, use_container_width=True)
            
            with col2:
                st.subheader("Basic Statistics")
                st.dataframe(df.describe(), use_container_width=True)
        
        with tab2:
            st.header("💬 Chat with your Data")
            
            init_chat_session()
            if st.session_state.df_context is None or not df.equals(st.session_state.df_context):
                st.session_state.df_context = df
                df_info = f"Dataset with {len(df)} rows and {len(df.columns)} columns. Columns: {', '.join(df.columns[:10])}"
            else:
                df_info = None
            
            if len(st.session_state.messages) == 0:
                st.markdown("### 💡 Suggested Questions")
                suggestions = get_suggested_questions(df)
                cols = st.columns(2)
                for i, suggestion in enumerate(suggestions):
                    with cols[i % 2]:
                        if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                            st.session_state.prompt_input = suggestion
                            st.rerun()
                st.markdown("---")
            
            # Chat history container
            chat_container = st.container()
            with chat_container:
                # Custom CSS for better chat styling
                st.markdown("""
                <style>
                    .stChatMessage {
                        padding: 10px;
                        margin: 5px 0;
                    }
                    div[data-testid="stChatMessageContent"] {
                        padding: 15px;
                        border-radius: 15px;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                # Display chat messages
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        content = message["content"]
                        
                        # Check if there's a chart reference in the content
                        if "[CHART:" in content:
                            # Extract text and chart ID
                            parts = content.split("[CHART:")
                            text_part = parts[0].strip()
                            chart_id = parts[1].replace("]", "").strip()
                            
                            # Display text
                            if text_part:
                                st.markdown(text_part)
                            
                            # Display chart if it exists
                            if 'charts' in st.session_state and chart_id in st.session_state.charts:
                                st.plotly_chart(st.session_state.charts[chart_id], use_container_width=True)
                        else:
                            st.markdown(content)
                        
                        if "timestamp" in message:
                            st.caption(message["timestamp"])
            
            # Input area at the bottom
            st.markdown("---")
            
            # Handle suggested question click
            prompt_value = st.session_state.get('prompt_input', '')
            
            # Chat input with Enter key support
            prompt = st.chat_input(
                "Ask a question about your data... (Press Enter to send)",
                key="chat_input_main"
            )
            
            # Handle suggested question click (fallback for button clicks)
            if prompt_value and not prompt:
                prompt = prompt_value
            
            if prompt:
                if 'prompt_input' in st.session_state:
                    del st.session_state.prompt_input
                
                timestamp = datetime.now().strftime("%H:%M")
                add_message("user", prompt, timestamp)
                logger.info(f"User query: {prompt}")
                
                context_prompt = create_context_prompt(prompt, df_info)
                
                with st.spinner("🤔 Thinking..."):
                    if st.session_state.agent:
                        try:
                            text_response, chart = st.session_state.agent.analyze_with_chart(df, prompt)
                            
                            if chart:
                                if 'charts' not in st.session_state:
                                    st.session_state.charts = {}
                                chart_id = f"chart_{len(st.session_state.messages)}"
                                st.session_state.charts[chart_id] = chart
                                response = text_response + f"\n\n[CHART:{chart_id}]"
                                logger.info(f"Generated response with chart for query: {prompt[:50]}...")
                            else:
                                response = text_response
                                logger.info(f"Generated text response for query: {prompt[:50]}...")
                        except Exception as e:
                            logger.error(f"Error processing query '{prompt}': {str(e)}")
                            response = "Sorry, I encountered an error processing your request. Please try again."
                    else:
                        response = "Please configure your API key (OpenAI or Google Gemini) in the sidebar to use the AI analyst."
                        logger.warning("User attempted query without configured API keys")
                
                add_message("assistant", response, timestamp)
                st.rerun()
            
            if len(st.session_state.messages) > 0:
                if st.button("🗑️ Clear Chat", key="clear_chat"):
                    st.session_state.messages = []
                    st.session_state.chat_context = []
                    logger.info("Chat history cleared by user")
                    st.rerun()
        
        with tab3:
            st.header("Data Visualizations")
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if numeric_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Distribution Plot")
                    selected_col = st.selectbox("Select column for distribution", numeric_cols, key="dist")
                    if selected_col:
                        fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Box Plot")
                    selected_col_box = st.selectbox("Select column for box plot", numeric_cols, key="box")
                    if selected_col_box:
                        fig = px.box(df, y=selected_col_box, title=f"Box Plot of {selected_col_box}")
                        st.plotly_chart(fig, use_container_width=True)
                
                if len(numeric_cols) > 1:
                    st.subheader("Correlation Heatmap")
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, 
                                  text_auto=True, 
                                  title="Correlation Matrix",
                                  color_continuous_scale='RdBu_r',
                                  zmin=-1, zmax=1)
                    st.plotly_chart(fig, use_container_width=True)
                
                if len(numeric_cols) >= 2:
                    st.subheader("Scatter Plot")
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("Select X axis", numeric_cols, key="scatter_x")
                    with col2:
                        y_col = st.selectbox("Select Y axis", numeric_cols, key="scatter_y")
                    
                    if x_col and y_col:
                        fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                        st.plotly_chart(fig, use_container_width=True)
            
            if categorical_cols:
                st.subheader("Categorical Analysis")
                selected_cat = st.selectbox("Select categorical column", categorical_cols)
                if selected_cat:
                    value_counts = df[selected_cat].value_counts().head(20)
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                               title=f"Top 20 values in {selected_cat}",
                               labels={'x': selected_cat, 'y': 'Count'})
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.header("AI-Generated Insights")
            
            if st.button("Generate Insights", type="primary"):
                with st.spinner("Generating insights..."):
                    try:
                        if st.session_state.agent:
                            insights = st.session_state.agent.get_insights(df)
                            logger.info("AI insights generated successfully")
                        else:
                            insights = []
                            summary = DataLoader.get_data_summary(df)
                            insights.append(f"📊 Dataset contains {summary['shape'][0]:,} rows and {summary['shape'][1]} columns")
                            insights.append(f"💾 Memory usage: {summary['memory_usage']:.2f} MB")
                            
                            null_cols = [col for col, count in summary['null_counts'].items() if count > 0]
                            if null_cols:
                                insights.append(f"⚠️ {len(null_cols)} columns have missing values")
                            
                            if summary['numeric_columns']:
                                insights.append(f"🔢 {len(summary['numeric_columns'])} numeric columns available for analysis")
                            
                            if summary['categorical_columns']:
                                insights.append(f"📝 {len(summary['categorical_columns'])} categorical columns detected")
                            
                            logger.info("Fallback insights generated")
                        
                        for insight in insights:
                            st.info(insight)
                    except Exception as e:
                        logger.error(f"Error generating insights: {str(e)}")
                        st.error("Failed to generate insights. Please try again.")
            
            st.subheader("Quick Analysis")
            
            questions = [
                "What are the main patterns in this data?",
                "Are there any correlations between variables?",
                "What are the outliers in the dataset?",
                "Summarize the key statistics",
                "What columns have missing data?"
            ]
            
            col1, col2 = st.columns(2)
            for i, question in enumerate(questions):
                with col1 if i % 2 == 0 else col2:
                    if st.button(question, key=f"q_{i}"):
                        with st.spinner("Analyzing..."):
                            try:
                                if st.session_state.agent:
                                    response = st.session_state.agent.analyze(df, question)
                                    st.write(response)
                                    logger.info(f"Quick analysis completed: {question}")
                                else:
                                    st.warning("Please configure your API key to use AI analysis")
                                    logger.warning("Quick analysis attempted without API key")
                            except Exception as e:
                                logger.error(f"Error in quick analysis '{question}': {str(e)}")
                                st.error("Analysis failed. Please try again.")

else:
    st.info("👈 Please upload a data file from the sidebar to get started!")
    
    st.subheader("Or try with sample data:")
    if st.button("Generate Sample Data"):
        import numpy as np
        
        try:
            np.random.seed(42)
            dates = pd.date_range(start='2024-01-01', periods=100)
            sample_df = pd.DataFrame({
                'Date': dates,
                'Sales': np.random.randint(1000, 5000, 100) + np.random.randn(100) * 500,
                'Units': np.random.randint(10, 100, 100),
                'Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 100),
                'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
                'Customer_Satisfaction': np.random.uniform(3.0, 5.0, 100)
            })
            
            st.session_state.df = sample_df
            logger.info("Sample data generated successfully")
            st.success("Sample data loaded! Refresh the page to analyze it.")
            st.rerun()
        except Exception as e:
            logger.error(f"Error generating sample data: {str(e)}")
            st.error("Failed to generate sample data. Please try again.")