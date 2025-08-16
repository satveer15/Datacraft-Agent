from langchain_experimental.agents import create_pandas_dataframe_agent  
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
import pandas as pd
import streamlit as st
from typing import Optional, Tuple, Any
import os
import re
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sys
import logging
from contextlib import redirect_stdout, redirect_stderr
from utils.chart_generator import ChartGenerator

logger = logging.getLogger(__name__)

class DataAnalystAgent:
    def __init__(self, openai_key: Optional[str] = None, gemini_key: Optional[str] = None):
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.gemini_key = gemini_key or os.getenv("GOOGLE_API_KEY")
        self.llm = None
        self.provider = None
        
        if self.openai_key and self.openai_key not in ["your_working_api_key_here", "", None]:
            try:
                self.llm = ChatOpenAI(
                    temperature=0,
                    model="gpt-3.5-turbo",
                    openai_api_key=self.openai_key
                )
                try:
                    self.llm.invoke("test")
                    self.provider = "OpenAI"
                    st.success("✅ Using OpenAI")
                    logger.info("OpenAI provider initialized successfully")
                except Exception as test_error:
                    if "quota" in str(test_error).lower() or "429" in str(test_error):
                        st.warning("⚠️ OpenAI quota exceeded, will try Gemini...")
                        logger.warning("OpenAI quota exceeded, falling back to Gemini")
                    else:
                        logger.error(f"OpenAI test failed: {str(test_error)}")
                    self.llm = None
            except Exception as e:
                logger.error(f"OpenAI initialization failed: {str(e)}")
                self.llm = None
        
        if not self.llm and self.gemini_key and self.gemini_key not in ["your_gemini_api_key_here", "", None]:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-exp",
                    temperature=0,
                    google_api_key=self.gemini_key
                )
                self.provider = "Google Gemini"
                st.success("✅ Using Google Gemini")
                logger.info("Google Gemini provider initialized successfully")
            except Exception as e:
                st.error(f"❌ Gemini initialization failed: {str(e)[:100]}")
                logger.error(f"Gemini initialization failed: {str(e)}")
                self.llm = None
        
        if not self.llm:
            st.error("❌ No AI provider available. Please configure valid API keys.")
            logger.error("No AI provider available - invalid API keys")
    
    def execute_code_safely(self, code: str, df: pd.DataFrame) -> Tuple[Any, str]:
        """Execute Python code safely and return result and output"""
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        
        namespace = {
            'df': df,
            'pd': pd,
            'px': px,
            'go': go,
            'plt': plt,
            'sns': sns,
            'np': np,
            'matplotlib': matplotlib,
            'print': print,
            '__builtins__': {
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'min': min,
                'max': max,
                'sum': sum,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'type': type,
                'isinstance': isinstance,
            }
        }
        
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        try:
            logger.debug(f"Executing code: {code[:100]}...")
            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                exec(code, namespace)
            
            # Check if a figure was created
            if 'fig' in namespace:
                fig = namespace['fig']
                # Validate that the figure has data
                if hasattr(fig, 'data') and len(fig.data) > 0:
                    # Check if the data traces have actual data points
                    has_data = False
                    for trace in fig.data:
                        if hasattr(trace, 'x') and hasattr(trace, 'y'):
                            if (trace.x is not None and len(trace.x) > 0) or (trace.y is not None and len(trace.y) > 0):
                                has_data = True
                                break
                        elif hasattr(trace, 'values') and trace.values is not None and len(trace.values) > 0:
                            has_data = True
                            break
                    
                    if has_data:
                        logger.info("Code execution successful - chart generated")
                        return fig, output_buffer.getvalue()
                    else:
                        logger.warning("Chart created but contains no data points")
                        return None, output_buffer.getvalue() + "\n⚠️ Figure created but contains no data points"
                else:
                    logger.warning("Chart created but has no data traces")
                    return None, output_buffer.getvalue() + "\n⚠️ Figure created but has no data traces"
            elif plt.get_fignums():
                plt.close('all')
                logger.warning("Matplotlib figure detected - should use Plotly instead")
                return None, output_buffer.getvalue() + "\n(Matplotlib figure created but not returned - use Plotly instead)"
            else:
                logger.info("Code execution successful - no chart generated")
                return None, output_buffer.getvalue()
                
        except Exception as e:
            error_msg = f"Error executing code: {str(e)}\n{error_buffer.getvalue()}"
            logger.error(f"Code execution failed: {str(e)}")
            return None, error_msg
    
    def analyze_with_chart(self, df: pd.DataFrame, query: str) -> Tuple[str, Optional[Any]]:
        """Analyze data and potentially generate a chart using AI-powered intent detection"""
        
        intent = self.classify_query_intent(df, query)
        logger.info(f"Query intent classified as: {intent}")
        
        if intent == "analysis":
            text_response = self.analyze(df, query)
            return f"**[Intent: Analysis]**\n\n{text_response}", None
        elif intent == "visualization":
            # Generate visualization
            text_response, chart = self.handle_visualization_request(df, query)
            return f"**[Intent: Visualization]**\n\n{text_response}", chart
        else:
            # Mixed or unclear intent - try both
            text_response, chart = self.handle_mixed_request(df, query)
            return f"**[Intent: Mixed]**\n\n{text_response}", chart
    
    def classify_query_intent(self, df: pd.DataFrame, query: str) -> str:
        """Use AI to classify the intent of the user query"""
        if not self.llm:
            # Fallback to simple heuristics if no AI available
            query_lower = query.lower()
            if any(word in query_lower for word in ['plot', 'chart', 'graph', 'visualize', 'show me a']):
                return "visualization"
            elif any(word in query_lower for word in ['insight', 'analysis', 'summary', 'pattern']):
                return "analysis"
            else:
                return "mixed"
        
        classification_prompt = f"""
        Given a user query about a dataset, classify the intent as one of:
        1. "analysis" - User wants insights, patterns, summaries, explanations about the data
        2. "visualization" - User wants to see charts, plots, graphs, visual representations
        3. "mixed" - User wants both analysis and visualization, or intent is unclear
        
        Dataset info: {len(df)} rows, {len(df.columns)} columns
        Columns: {', '.join(df.columns[:10])}
        
        User query: "{query}"
        
        Respond with only one word: analysis, visualization, or mixed
        """
        
        try:
            response = self.llm.invoke(classification_prompt)
            intent = response.content.strip().lower() if hasattr(response, 'content') else str(response).strip().lower()
            
            if intent in ['analysis', 'visualization', 'mixed']:
                return intent
            else:
                return "mixed"  # Default fallback
                
        except Exception as e:
            # Fallback classification
            query_lower = query.lower()
            if any(word in query_lower for word in ['plot', 'chart', 'visualize', 'show']):
                return "visualization"
            else:
                return "analysis"
    
    def handle_visualization_request(self, df: pd.DataFrame, query: str) -> Tuple[str, Optional[Any]]:
        """Handle requests that are specifically for visualization - prioritize hybrid approach"""
        
        # Step 1: Try hybrid AI approach first (sample for code generation, execute on full dataset)
        chart_code = self.generate_chart_code(df, query)
        
        if chart_code:
            # Execute the code to generate the chart on full dataset
            fig, output = self.execute_code_safely(chart_code, df)
            
            if fig:
                text_response = f"""**[Using {self.provider} - Hybrid Approach]**

✅ **Generated visualization with complete dataset!**

**Process:**
- AI analyzed: 100 sample rows (minimal tokens)
- Chart generated with: {len(df):,} rows (full dataset)
- Token savings: ~95% vs sending full data

**Code executed:**
```python
{chart_code}
```"""
                if output:
                    text_response += f"\n\n**Debug output:**\n```\n{output}\n```"
                return text_response, fig
            else:
                text_response = f"**[Using {self.provider}]**\n\n⚠️ **AI code failed, trying local generation...**\n\n**Code attempted:**\n```python\n{chart_code}\n```"
                if output:
                    text_response += f"\n\n**Debug output:**\n```\n{output}\n```"
        else:
            text_response = f"**[Using {self.provider}]**\n\n⚠️ **AI code generation failed, trying local generation...**"
        
        # Step 2: Fallback to local chart generation
        local_chart = ChartGenerator.generate_chart_from_query(df, query)
        
        if local_chart:
            chart_info = ChartGenerator.parse_chart_request(query, df)
            mentioned_cols = chart_info.get('columns', [])
            chart_type = chart_info.get('chart_type', 'auto')
            
            text_response += f"""

✅ **Fallback: Local chart generator used**

**Chart Details:**
- Chart type: {chart_type}
- Detected columns: {mentioned_cols}
- Using full dataset: {df.shape}
- Token usage: 0 (local fallback)"""
            return text_response, local_chart
        
        # Step 3: Enhanced local chart generation with query analysis
        enhanced_chart = self.generate_enhanced_local_chart(df, query)
        
        if enhanced_chart:
            text_response += f"\n\n✅ **Enhanced local chart generated with full dataset ({len(df):,} rows)**"
            return text_response, enhanced_chart
        else:
            # Final fallback
            text_response += f"\n\n❌ **Could not generate visualization**\n\n**Available columns:** {list(df.columns)}\n**Data shape:** {df.shape}"
            return text_response, None
    
    def handle_mixed_request(self, df: pd.DataFrame, query: str) -> Tuple[str, Optional[Any]]:
        """Handle requests that need both analysis and visualization"""
        # Try to generate both analysis and chart
        analysis_text = self.analyze(df, query)
        
        # Try to generate a chart
        chart_code = self.generate_chart_code(df, query)
        chart = None
        chart_info = ""
        
        if chart_code:
            chart, output = self.execute_code_safely(chart_code, df)
            if chart:
                chart_info = f"""

**Visualization Generated (Hybrid Approach):**
- AI analyzed: 100 sample rows 
- Chart uses: {len(df):,} full dataset rows
- Code executed:
```python
{chart_code}
```"""
                if output:
                    chart_info += f"\n\n**Debug Output:**\n```\n{output}\n```"
            else:
                chart_info = f"\n\n**Attempted Code (no chart generated):**\n```python\n{chart_code}\n```"
                if output:
                    chart_info += f"\n\n**Debug Output:**\n```\n{output}\n```"
        
        if not chart:
            # Try fallback chart generator
            chart = ChartGenerator.generate_chart_from_query(df, query)
            if chart:
                chart_info += f"\n\n**Used fallback chart generator** - Available columns: {list(df.columns)}"
        
        # Combine analysis and chart info
        combined_text = analysis_text + chart_info
        
        return combined_text, chart
    
    def generate_chart_code(self, df: pd.DataFrame, query: str) -> Optional[str]:
        """Generate chart code using AI with small sample, but code will run on full dataset"""
        if not self.llm:
            return None
        
        # Create small sample for AI to understand data structure (minimal tokens)
        sample_size = min(100, len(df))  # Very small sample just for structure
        sample_df = df.sample(n=sample_size, random_state=42)
        
        # Get column types for better context
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        prompt = f"""
        Create ONE SINGLE plotly chart for a DataFrame 'df'. Here's a small sample for structure understanding:
        
        SAMPLE DATA (structure only):
        {sample_df.head(5).to_string()}
        
        FULL DATASET INFO:
        - Total rows: {len(df):,}
        - Columns: {list(df.columns)}
        - Numeric: {numeric_cols}
        - Categorical: {categorical_cols}
        
        USER REQUEST: "{query}"
        
        CRITICAL REQUIREMENTS:
        1. Create ONLY ONE chart (not multiple)
        2. Create variable 'fig' with plotly express figure
        3. Use DataFrame 'df' (full dataset, not sample)
        4. Use ONLY exact column names: {list(df.columns)}
        5. NO function definitions, NO imports, NO sample data creation
        6. Handle missing values gracefully
        7. Return ONLY direct executable code
        8. Add debugging prints
        
        FORBIDDEN:
        - Multiple charts (fig1, fig2, etc.)
        - Function definitions (def analyze_something)
        - Creating sample data or DataFrames
        - Using matplotlib, seaborn, or fig.show()
        - Any code after creating the 'fig' variable except fig.update_layout()
        
        REQUIRED FORMAT (choose ONE chart type):
        print(f"Full dataset: {{df.shape}}")
        print(f"Using columns: x='column1', y='column2'")
        filtered_df = df[df['column'] == 'value']  # Optional filtering
        fig = px.scatter(filtered_df, x='column1', y='column2', title='Chart Title')
        fig.update_layout(height=500)
        print(f"Chart created with {{len(filtered_df)}} data points")
        
        CRITICAL SYNTAX RULES:
        - Each line must be complete and syntactically correct
        - All parentheses must be properly closed
        - No line breaks within function calls
        - No trailing commas before closing parentheses
        """
        
        try:
            response = self.llm.invoke(prompt)
            
            # Extract code from response
            code = response.content if hasattr(response, 'content') else str(response)
            
            # Clean up the code - extract just the Python code
            if '```python' in code:
                code = code.split('```python')[1].split('```')[0]
            elif '```' in code:
                code = code.split('```')[1].split('```')[0]
            
            # Aggressively clean and filter code
            lines = code.split('\n')
            cleaned_lines = []
            skip_function = False
            skip_main = False
            fig_found = False
            
            for line in lines:
                stripped_line = line.strip()
                original_line = line
                
                # Skip imports
                if stripped_line.startswith(('import ', 'from ')):
                    continue
                
                # Skip function definitions and their content
                if stripped_line.startswith('def ') or '"""' in stripped_line:
                    skip_function = True
                    continue
                elif skip_function and (line.startswith('    ') or stripped_line == '' or stripped_line.startswith('"""')):
                    continue  # Skip function body and docstrings
                elif skip_function and not line.startswith('    '):
                    skip_function = False  # End of function
                
                # Skip if __name__ == '__main__' blocks
                if stripped_line.startswith("if __name__ == '__main__'"):
                    skip_main = True
                    continue
                elif skip_main:
                    continue
                
                # Skip problematic patterns
                if any(pattern in stripped_line for pattern in [
                    'pd.DataFrame(', 'data = {', 'df = pd.concat', 
                    'fig.show()', 'plt.', 'sns.', 'matplotlib',
                    'def ', 'return', 'Args:', 'analyze_'
                ]):
                    continue
                
                # Skip comments and docstrings
                if stripped_line.startswith(('#', '"""', "'''")):
                    continue
                
                # Only allow lines after we have basic prints or actual plotting code
                if stripped_line and not stripped_line.startswith(' '):
                    # Keep print statements and variable assignments
                    if any(pattern in stripped_line for pattern in [
                        'print(', 'fig = px.', 'fig.update_layout', 
                        '_df = df[', 'filtered_df =', 'df_'
                    ]):
                        cleaned_lines.append(original_line)
                    # Keep simple variable assignments for filtering
                    elif '=' in stripped_line and 'df[' in stripped_line:
                        cleaned_lines.append(original_line)
                elif stripped_line and line.startswith(' '):
                    # Keep indented lines only if they're simple and safe
                    if any(safe in stripped_line for safe in ['print(', 'df[', '==']):
                        cleaned_lines.append(original_line)
            
            code = '\n'.join(cleaned_lines)
            
            # Fix common syntax issues
            # Fix incomplete px.scatter calls
            if 'fig = px.scatter(' in code and 'fig.update_layout' in code:
                lines = code.split('\n')
                fixed_lines = []
                for i, line in enumerate(lines):
                    if 'fig = px.scatter(' in line and not line.rstrip().endswith(')'):
                        # Find the next line to see if it's fig.update_layout
                        if i + 1 < len(lines) and 'fig.update_layout' in lines[i + 1]:
                            # Close the px.scatter call
                            line = line.rstrip().rstrip(',') + ')'
                    fixed_lines.append(line)
                code = '\n'.join(fixed_lines)
            
            # Fix common indentation issues
            if 'if df.shape[0] > 50000:' in code and 'df = df.sample' in code:
                code = code.replace(
                    'if df.shape[0] > 50000:\ndf = df.sample',
                    'if df.shape[0] > 50000:\n    df = df.sample'
                )
            
            # Validate syntax by checking for common patterns
            if 'fig = px.' not in code:
                return None
                
            # Check for incomplete function calls
            px_line = None
            for line in code.split('\n'):
                if 'fig = px.' in line:
                    px_line = line.strip()
                    break
            
            if px_line:
                # Count parentheses to ensure they're balanced
                open_parens = px_line.count('(')
                close_parens = px_line.count(')')
                if open_parens != close_parens:
                    # Try to fix by adding missing closing parenthesis
                    if open_parens > close_parens:
                        code = code.replace(px_line, px_line + ')')
            
            # Final validation
            if 'fig' in code and ('px.' in code or 'go.' in code):
                return code.strip()
            else:
                return None
                
        except Exception as e:
            st.warning(f"Could not generate chart code: {str(e)[:100]}")
            return None
    
    def analyze(self, df: pd.DataFrame, query: str) -> str:
        """Analyze data using smart sampling and local processing + AI interpretation"""
        try:
            if not self.llm:
                return "Please configure your API key (OpenAI or Google Gemini) to use the AI analyst."
            
            provider_info = f"**[Using {self.provider}]**\n\n"
            
            # Generate comprehensive local insights first
            data_summary = self.generate_comprehensive_data_summary(df, query)
            
            # Use AI to interpret the summary instead of processing raw data
            interpretation = self.get_ai_interpretation(data_summary, query)
            
            return provider_info + interpretation
                    
        except Exception as e:
            # Fallback to local insights only
            return f"**[Using Local Analysis]**\n\n" + self.generate_fallback_insights(df, query)
    
    def create_agent(self, df: pd.DataFrame):
        """Create a pandas dataframe agent"""
        if not self.llm:
            return None
        
        try:
            # Use appropriate agent type based on provider
            if self.provider == "Google Gemini":
                agent = create_pandas_dataframe_agent(
                    self.llm,
                    df,
                    verbose=False,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    allow_dangerous_code=True,
                    max_iterations=3,
                    early_stopping_method="generate"
                )
            else:
                agent = create_pandas_dataframe_agent(
                    self.llm,
                    df,
                    verbose=False,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    allow_dangerous_code=True
                )
            return agent
        except Exception as e:
            st.error(f"Failed to create agent: {str(e)[:200]}")
            return None
    
    def get_insights(self, df: pd.DataFrame) -> list:
        """Generate basic insights about the dataset"""
        insights = []
        
        # Basic statistics
        insights.append(f"📊 Dataset has {len(df):,} rows and {len(df.columns)} columns")
        
        # Missing values
        missing = df.isnull().sum()
        if missing.any():
            top_missing = missing[missing > 0].head(3)
            for col, count in top_missing.items():
                insights.append(f"⚠️ Column '{col}' has {count:,} missing values ({count/len(df)*100:.1f}%)")
        
        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]:
                try:
                    insights.append(f"📈 '{col}' ranges from {df[col].min():.2f} to {df[col].max():.2f}")
                except:
                    pass
        
        # Categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            for col in cat_cols[:3]:
                unique_count = df[col].nunique()
                insights.append(f"🏷️ '{col}' has {unique_count:,} unique values")
        
        return insights
    
    def generate_fallback_insights(self, df: pd.DataFrame, query: str) -> str:
        """Generate insights without AI when the agent fails"""
        insights = []
        
        # Basic dataset info
        insights.append(f"📊 **Dataset Overview**: {len(df):,} rows, {len(df.columns)} columns")
        
        # Column analysis
        numeric_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 0:
            insights.append(f"🔢 **Numeric columns** ({len(numeric_cols)}): {', '.join(numeric_cols[:5])}")
            
            # Find interesting numeric patterns
            for col in numeric_cols[:3]:
                try:
                    col_stats = df[col].describe()
                    if col_stats['std'] > 0:
                        cv = col_stats['std'] / col_stats['mean'] if col_stats['mean'] != 0 else 0
                        if cv > 1:
                            insights.append(f"⚡ **{col}** has high variability (CV: {cv:.2f}) - ranges from {col_stats['min']:.2f} to {col_stats['max']:.2f}")
                        
                        # Check for outliers
                        q1, q3 = col_stats['25%'], col_stats['75%']
                        iqr = q3 - q1
                        outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)]
                        if len(outliers) > 0:
                            insights.append(f"🎯 **{col}** has {len(outliers):,} potential outliers ({len(outliers)/len(df)*100:.1f}%)")
                except:
                    pass
        
        if len(cat_cols) > 0:
            insights.append(f"🏷️ **Categorical columns** ({len(cat_cols)}): {', '.join(cat_cols[:5])}")
            
            # Find interesting categorical patterns
            for col in cat_cols[:3]:
                try:
                    value_counts = df[col].value_counts()
                    if len(value_counts) > 1:
                        top_category = value_counts.index[0]
                        top_pct = value_counts.iloc[0] / len(df) * 100
                        insights.append(f"🏆 **{col}**: '{top_category}' dominates with {top_pct:.1f}% ({value_counts.iloc[0]:,} records)")
                        
                        if len(value_counts) > 10:
                            insights.append(f"🌈 **{col}** has high diversity: {len(value_counts):,} unique values")
                except:
                    pass
        
        # Missing data analysis
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            total_missing = missing.sum()
            insights.append(f"⚠️ **Missing data**: {total_missing:,} missing values across {len(missing)} columns")
            worst_col = missing.index[0]
            worst_pct = missing.iloc[0] / len(df) * 100
            insights.append(f"💔 **{worst_col}** has the most missing data: {worst_pct:.1f}%")
        
        # Correlations for numeric data
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr()
                # Find highest correlations
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:  # Strong correlation
                            corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
                
                if corr_pairs:
                    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                    col1, col2, corr = corr_pairs[0]
                    insights.append(f"🔗 **Strong correlation**: {col1} ↔ {col2} ({corr:.3f})")
            except:
                pass
        
        # Query-specific insights
        query_lower = query.lower()
        if 'crazy' in query_lower or 'interesting' in query_lower:
            insights.append("🎪 **Crazy finding**: This dataset is pretty wild with its scale and complexity!")
            
            # Find some "crazy" stats
            if len(df) > 100000:
                insights.append(f"🤯 **Scale shock**: With {len(df):,} rows, this dataset is MASSIVE!")
            
            if len(cat_cols) > 0:
                for col in cat_cols[:2]:
                    try:
                        unique_ratio = df[col].nunique() / len(df)
                        if unique_ratio > 0.8:
                            insights.append(f"🦄 **{col}** is almost entirely unique values ({unique_ratio:.1%}) - very high cardinality!")
                        elif unique_ratio < 0.01:
                            insights.append(f"🎯 **{col}** has very low diversity ({unique_ratio:.1%}) - highly concentrated data!")
                    except:
                        pass
        
        return "\n\n".join(insights) if insights else f"Dataset analysis: {len(df):,} rows and {len(df.columns)} columns. Enable AI for deeper insights."
    
    def generate_comprehensive_data_summary(self, df: pd.DataFrame, query: str) -> dict:
        """Generate comprehensive statistical summary without sending raw data"""
        
        # Get column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Smart sampling based on query and dataset size
        sample_df = self.get_smart_sample(df, query)
        
        summary = {
            "dataset_info": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "sample_size": len(sample_df),
                "columns": list(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2
            },
            
            "column_types": {
                "numeric": numeric_cols,
                "categorical": categorical_cols,
                "datetime": datetime_cols
            },
            
            "statistical_summary": {},
            "categorical_summary": {},
            "correlations": {},
            "missing_data": df.isnull().sum().to_dict(),
            "query_specific_insights": {}
        }
        
        # Statistical summary for numeric columns
        if numeric_cols:
            stats_df = df[numeric_cols].describe()
            summary["statistical_summary"] = stats_df.to_dict()
            
            # Correlations (only for numeric columns to save space)
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                # Only include strong correlations (>0.5) to reduce data
                strong_corrs = {}
                for i, col1 in enumerate(corr_matrix.columns):
                    for j, col2 in enumerate(corr_matrix.columns):
                        if i < j and abs(corr_matrix.iloc[i, j]) > 0.5:
                            strong_corrs[f"{col1}_vs_{col2}"] = round(corr_matrix.iloc[i, j], 3)
                summary["correlations"] = strong_corrs
        
        # Categorical summary (top values only)
        if categorical_cols:
            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                value_counts = df[col].value_counts()
                summary["categorical_summary"][col] = {
                    "unique_values": len(value_counts),
                    "top_10_values": value_counts.head(10).to_dict(),
                    "null_percentage": (df[col].isnull().sum() / len(df)) * 100
                }
        
        # Query-specific insights
        summary["query_specific_insights"] = self.extract_query_relevant_data(df, query, sample_df)
        
        return summary
    
    def get_smart_sample(self, df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Get intelligent sample based on query type and dataset characteristics"""
        
        # Determine sample size based on dataset size
        if len(df) <= 5000:
            return df  # Use full dataset for small data
        elif len(df) <= 50000:
            sample_size = 3000
        else:
            sample_size = 5000
        
        query_lower = query.lower()
        
        # Query-specific sampling strategies
        if any(word in query_lower for word in ['trend', 'time', 'over time', 'temporal']):
            # For time-based queries, try to sample across time periods
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                return df.sort_values(datetime_cols[0]).iloc[::max(1, len(df)//sample_size)]
        
        elif any(word in query_lower for word in ['category', 'group', 'by']):
            # For categorical analysis, ensure representation from all groups
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                return df.groupby(categorical_cols[0], observed=True).apply(
                    lambda x: x.sample(min(len(x), sample_size//df[categorical_cols[0]].nunique()))
                ).reset_index(drop=True)
        
        # Default: stratified random sampling
        return df.sample(n=min(sample_size, len(df)), random_state=42)
    
    def extract_query_relevant_data(self, df: pd.DataFrame, query: str, sample_df: pd.DataFrame) -> dict:
        """Extract data specifically relevant to the user's query"""
        
        query_lower = query.lower()
        insights = {}
        
        # Extract column names mentioned in query
        mentioned_cols = []
        for col in df.columns:
            if col.lower() in query_lower or any(word in col.lower() for word in query_lower.split()):
                mentioned_cols.append(col)
        
        if mentioned_cols:
            insights["mentioned_columns"] = mentioned_cols
            
            # Get specific stats for mentioned columns
            for col in mentioned_cols[:3]:  # Limit to first 3 mentioned columns
                if col in df.select_dtypes(include=['number']).columns:
                    insights[f"{col}_stats"] = {
                        "mean": df[col].mean(),
                        "median": df[col].median(),
                        "std": df[col].std(),
                        "min": df[col].min(),
                        "max": df[col].max(),
                        "outliers_count": len(df[(df[col] < df[col].quantile(0.25) - 1.5*(df[col].quantile(0.75)-df[col].quantile(0.25))) | 
                                                (df[col] > df[col].quantile(0.75) + 1.5*(df[col].quantile(0.75)-df[col].quantile(0.25)))])
                    }
                elif col in df.select_dtypes(include=['object', 'category']).columns:
                    insights[f"{col}_distribution"] = df[col].value_counts().head(10).to_dict()
        
        # Query type insights
        if any(word in query_lower for word in ['pattern', 'trend', 'insight', 'analyze']):
            insights["patterns_detected"] = self.detect_patterns(sample_df)
        
        return insights
    
    def detect_patterns(self, df: pd.DataFrame) -> dict:
        """Detect interesting patterns in the sample data"""
        patterns = {}
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Detect skewness in numeric columns
        if numeric_cols:
            from scipy import stats
            skewed_cols = []
            for col in numeric_cols:
                try:
                    skewness = stats.skew(df[col].dropna())
                    if abs(skewness) > 1:  # Highly skewed
                        skewed_cols.append({"column": col, "skewness": round(skewness, 2)})
                except:
                    pass
            if skewed_cols:
                patterns["highly_skewed_columns"] = skewed_cols
        
        # Detect imbalanced categories
        if categorical_cols:
            imbalanced_cats = []
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                if len(value_counts) > 1:
                    dominance = value_counts.iloc[0] / len(df)
                    if dominance > 0.8:  # Highly imbalanced
                        imbalanced_cats.append({
                            "column": col,
                            "dominant_value": value_counts.index[0],
                            "dominance_pct": round(dominance * 100, 1)
                        })
            if imbalanced_cats:
                patterns["imbalanced_categories"] = imbalanced_cats
        
        return patterns
    
    def get_ai_interpretation(self, data_summary: dict, query: str) -> str:
        """Use AI to interpret the data summary instead of processing raw data"""
        
        interpretation_prompt = f"""
        Analyze this data summary and provide insights for the user query.
        
        USER QUERY: "{query}"
        
        DATA SUMMARY:
        Dataset: {data_summary['dataset_info']['total_rows']:,} rows, {data_summary['dataset_info']['total_columns']} columns
        
        Columns: {data_summary['column_types']}
        
        Statistical Summary: {data_summary['statistical_summary']}
        
        Categorical Data: {data_summary['categorical_summary']}
        
        Strong Correlations: {data_summary['correlations']}
        
        Missing Data: {data_summary['missing_data']}
        
        Query-Specific Insights: {data_summary['query_specific_insights']}
        
        Please provide a comprehensive analysis addressing the user's query. Include:
        1. Direct answer to their question
        2. Key patterns and insights from the data
        3. Notable findings about distributions, correlations, or trends
        4. Data quality observations
        5. Actionable recommendations
        
        Keep the response informative but concise.
        """
        
        try:
            response = self.llm.invoke(interpretation_prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            # Fallback to structured summary if AI fails
            return self.format_summary_insights(data_summary, query)
    
    def format_summary_insights(self, data_summary: dict, query: str) -> str:
        """Format data summary into readable insights when AI interpretation fails"""
        
        insights = []
        dataset_info = data_summary['dataset_info']
        
        insights.append(f"📊 **Dataset Overview**: {dataset_info['total_rows']:,} rows, {dataset_info['total_columns']} columns")
        insights.append(f"💾 **Memory Usage**: {dataset_info['memory_usage_mb']:.1f} MB")
        
        # Column types
        col_types = data_summary['column_types']
        if col_types['numeric']:
            insights.append(f"🔢 **Numeric columns** ({len(col_types['numeric'])}): {', '.join(col_types['numeric'][:5])}")
        if col_types['categorical']:
            insights.append(f"🏷️ **Categorical columns** ({len(col_types['categorical'])}): {', '.join(col_types['categorical'][:5])}")
        
        # Statistical insights
        if data_summary['statistical_summary']:
            insights.append("📈 **Key Statistics**:")
            for col, stats in list(data_summary['statistical_summary'].items())[:3]:
                mean_val = stats.get('mean', 0)
                std_val = stats.get('std', 0)
                cv = std_val / mean_val if mean_val != 0 else 0
                insights.append(f"   • {col}: avg={mean_val:.2f}, range=[{stats.get('min', 0):.2f}, {stats.get('max', 0):.2f}], variability={'high' if cv > 1 else 'low'}")
        
        # Correlations
        if data_summary['correlations']:
            insights.append("🔗 **Strong Correlations**:")
            for pair, corr in list(data_summary['correlations'].items())[:3]:
                insights.append(f"   • {pair.replace('_vs_', ' ↔ ')}: {corr}")
        
        # Missing data
        missing_data = {k: v for k, v in data_summary['missing_data'].items() if v > 0}
        if missing_data:
            total_missing = sum(missing_data.values())
            insights.append(f"⚠️ **Missing Data**: {total_missing:,} missing values in {len(missing_data)} columns")
        
        # Query-specific insights
        query_insights = data_summary['query_specific_insights']
        if query_insights.get('mentioned_columns'):
            insights.append(f"🎯 **Query-relevant columns**: {', '.join(query_insights['mentioned_columns'])}")
        
        return "\n\n".join(insights)
    
    def generate_chart_metadata(self, df: pd.DataFrame) -> dict:
        """Generate minimal metadata for chart generation to reduce token usage"""
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Suggest reasonable defaults for common chart types
        suggested_x = numeric_cols[0] if numeric_cols else categorical_cols[0] if categorical_cols else df.columns[0]
        suggested_y = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0] if numeric_cols else categorical_cols[0] if categorical_cols else df.columns[1] if len(df.columns) > 1 else df.columns[0]
        suggested_color = categorical_cols[0] if categorical_cols else None
        
        return {
            "columns": list(df.columns),
            "column_info": {
                "numeric": numeric_cols[:5],  # Limit to first 5 for token efficiency
                "categorical": categorical_cols[:5],
                "datetime": datetime_cols[:3]
            },
            "suggested_x": suggested_x,
            "suggested_y": suggested_y,
            "suggested_color": suggested_color,
            "data_types": {col: str(df[col].dtype) for col in df.columns[:10]}  # Only first 10 columns
        }
    
    def generate_enhanced_local_chart(self, df: pd.DataFrame, query: str) -> Optional[Any]:
        """Enhanced local chart generation with better query understanding"""
        
        # Extract specific requirements from query
        query_lower = query.lower()
        
        # Find mentioned columns with fuzzy matching
        mentioned_cols = []
        for col in df.columns:
            col_words = col.lower().replace('_', ' ').split()
            query_words = query_lower.replace('_', ' ').split()
            
            # Direct match
            if col.lower() in query_lower:
                mentioned_cols.append(col)
            # Fuzzy word matching
            elif any(word in query_lower for word in col_words if len(word) > 2):
                mentioned_cols.append(col)
            # Check if query words are in column name
            elif any(word in col.lower() for word in query_words if len(word) > 2):
                mentioned_cols.append(col)
        
        # Remove duplicates
        mentioned_cols = list(dict.fromkeys(mentioned_cols))
        
        if not mentioned_cols:
            return None
        
        # Determine chart type and create appropriate visualization
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Determine best chart type based on query and data types
            if any(word in query_lower for word in ['scatter', 'correlation', 'vs', 'against']):
                return self.create_scatter_plot(df, mentioned_cols, numeric_cols, categorical_cols)
            elif any(word in query_lower for word in ['bar', 'count', 'frequency', 'distribution']):
                return self.create_bar_plot(df, mentioned_cols, categorical_cols, numeric_cols)
            elif any(word in query_lower for word in ['line', 'trend', 'time', 'over']):
                return self.create_line_plot(df, mentioned_cols, numeric_cols)
            elif any(word in query_lower for word in ['histogram', 'distribution']):
                return self.create_histogram(df, mentioned_cols, numeric_cols)
            else:
                # Auto-detect based on column types
                return self.auto_create_chart(df, mentioned_cols, numeric_cols, categorical_cols)
                
        except Exception as e:
            return None
    
    def create_scatter_plot(self, df: pd.DataFrame, mentioned_cols: list, numeric_cols: list, categorical_cols: list) -> Optional[Any]:
        """Create scatter plot with full dataset"""
        try:
            # Find x and y columns
            x_col = None
            y_col = None
            color_col = None
            
            # Use mentioned columns that are numeric
            numeric_mentioned = [col for col in mentioned_cols if col in numeric_cols]
            categorical_mentioned = [col for col in mentioned_cols if col in categorical_cols]
            
            if len(numeric_mentioned) >= 2:
                x_col, y_col = numeric_mentioned[0], numeric_mentioned[1]
            elif len(numeric_mentioned) == 1:
                x_col = numeric_mentioned[0]
                y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            else:
                return None
            
            # Add color if categorical column is mentioned
            if categorical_mentioned:
                color_col = categorical_mentioned[0]
            elif categorical_cols:
                color_col = categorical_cols[0]
            
            # Create the plot with full dataset
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                           title=f'{y_col} vs {x_col}' + (f' by {color_col}' if color_col else ''),
                           labels={x_col: x_col.replace('_', ' ').title(),
                                  y_col: y_col.replace('_', ' ').title()})
            fig.update_layout(height=500)
            return fig
            
        except Exception:
            return None
    
    def create_bar_plot(self, df: pd.DataFrame, mentioned_cols: list, categorical_cols: list, numeric_cols: list) -> Optional[Any]:
        """Create bar plot with full dataset"""
        try:
            x_col = None
            y_col = None
            
            # Find categorical and numeric columns
            categorical_mentioned = [col for col in mentioned_cols if col in categorical_cols]
            numeric_mentioned = [col for col in mentioned_cols if col in numeric_cols]
            
            if categorical_mentioned:
                x_col = categorical_mentioned[0]
                if numeric_mentioned:
                    # Aggregated bar chart
                    y_col = numeric_mentioned[0]
                    agg_df = df.groupby(x_col)[y_col].sum().reset_index()
                    fig = px.bar(agg_df, x=x_col, y=y_col,
                               title=f'{y_col.replace("_", " ").title()} by {x_col.replace("_", " ").title()}')
                else:
                    # Count bar chart
                    value_counts = df[x_col].value_counts()
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                               title=f'{x_col.replace("_", " ").title()} Distribution',
                               labels={'x': x_col.replace('_', ' ').title(), 'y': 'Count'})
            else:
                return None
            
            fig.update_layout(height=500)
            return fig
            
        except Exception:
            return None
    
    def create_line_plot(self, df: pd.DataFrame, mentioned_cols: list, numeric_cols: list) -> Optional[Any]:
        """Create line plot with full dataset"""
        try:
            # Look for datetime column or use first mentioned column as x
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            x_col = datetime_cols[0] if datetime_cols else mentioned_cols[0]
            y_col = [col for col in mentioned_cols if col in numeric_cols][0] if any(col in numeric_cols for col in mentioned_cols) else numeric_cols[0]
            
            # Sort by x column for proper line chart
            df_sorted = df.sort_values(x_col)
            
            fig = px.line(df_sorted, x=x_col, y=y_col,
                         title=f'{y_col.replace("_", " ").title()} over {x_col.replace("_", " ").title()}')
            fig.update_layout(height=500)
            return fig
            
        except Exception:
            return None
    
    def create_histogram(self, df: pd.DataFrame, mentioned_cols: list, numeric_cols: list) -> Optional[Any]:
        """Create histogram with full dataset"""
        try:
            # Use first numeric mentioned column
            col = [col for col in mentioned_cols if col in numeric_cols][0]
            
            fig = px.histogram(df, x=col, 
                             title=f'Distribution of {col.replace("_", " ").title()}')
            fig.update_layout(height=500)
            return fig
            
        except Exception:
            return None
    
    def auto_create_chart(self, df: pd.DataFrame, mentioned_cols: list, numeric_cols: list, categorical_cols: list) -> Optional[Any]:
        """Auto-create appropriate chart based on column types"""
        try:
            numeric_mentioned = [col for col in mentioned_cols if col in numeric_cols]
            categorical_mentioned = [col for col in mentioned_cols if col in categorical_cols]
            
            if len(numeric_mentioned) >= 2:
                # Scatter plot for two numeric columns
                return self.create_scatter_plot(df, mentioned_cols, numeric_cols, categorical_cols)
            elif len(numeric_mentioned) == 1 and len(categorical_mentioned) >= 1:
                # Bar plot for numeric vs categorical
                return self.create_bar_plot(df, mentioned_cols, categorical_cols, numeric_cols)
            elif len(categorical_mentioned) >= 1:
                # Count plot for categorical
                return self.create_bar_plot(df, mentioned_cols, categorical_cols, numeric_cols)
            elif len(numeric_mentioned) == 1:
                # Histogram for single numeric
                return self.create_histogram(df, mentioned_cols, numeric_cols)
            else:
                return None
                
        except Exception:
            return None