# 🛠️ DataCraft SuperTool

> **AI-Powered Data Analytics Assistant** - Transform your data analysis through natural language conversations

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.32.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 What is DataCraft SuperTool?

DataCraft SuperTool is a sophisticated AI-powered analytics platform that makes data analysis accessible to everyone. Simply upload your data and ask questions in plain English - the AI will analyze your data, generate insights, and create beautiful interactive visualizations.

### ✨ Key Features

- 🤖 **Conversational AI**: Ask questions like "What are the sales trends?" or "Show me outliers"
- 📊 **Smart Visualizations**: AI automatically creates appropriate charts and graphs
- 🔄 **Multi-AI Support**: Uses OpenAI GPT and Google Gemini with automatic failover
- 💰 **Cost-Optimized**: 95% reduction in AI costs through intelligent sampling
- 🛡️ **Secure**: Sandboxed code execution with comprehensive safety measures
- 📁 **Multi-Format**: Supports CSV, Excel, and JSON files

### 🚀 Quick Demo

```
You: "Show me sales trends by region"
AI: "I can see an interesting pattern in your data..."
    [Creates interactive chart showing regional sales over time]

You: "What are the top performing products?"
AI: "Based on your data, here are the insights..."
    [Generates bar chart with product performance analysis]
```

## 🛠️ Quick Start

### Prerequisites

- Python 3.8 or higher
- At least one AI API key (OpenAI or Google Gemini)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/datacraft-supertool.git
   cd datacraft-supertool
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=sk-your-openai-key-here
   GOOGLE_API_KEY=your-gemini-key-here
   ```
   
   Or enter them directly in the app sidebar.

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   
   Navigate to `http://localhost:8501` and start analyzing your data!

## 🔑 Getting API Keys

### OpenAI API Key
1. Visit [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new secret key
5. Copy the key (starts with `sk-`)

### Google Gemini API Key
1. Visit [makersuite.google.com](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

## 💡 Usage Examples

### Basic Data Analysis
1. Upload your CSV, Excel, or JSON file
2. Ask questions like:
   - "What insights do you see in this data?"
   - "Show me the distribution of sales"
   - "Create a correlation matrix"
   - "Find any outliers or anomalies"

### Advanced Visualizations
- "Create a scatter plot of price vs quantity"
- "Show me sales trends over time"
- "Compare performance by category"
- "Generate a pie chart of market share"

### Business Intelligence
- "Which products are performing best?"
- "What are the seasonal trends?"
- "Identify the top customers by revenue"
- "Show me year-over-year growth"

## 🏗️ Architecture

### Core Components

```
datacraft-supertool/
├── app.py                    # Main Streamlit application
├── agents/
│   └── data_analyst.py       # AI agent with multi-provider support
├── utils/
│   ├── data_loader.py        # File processing utilities
│   ├── chat_ui.py            # Conversational interface
│   └── chart_generator.py    # Local visualization fallback
└── data/
    └── sample_sales_data.csv # Example dataset
```

### Key Innovations

- **Hybrid AI Processing**: Seamlessly switches between AI providers
- **Token Optimization**: Uses smart sampling to reduce costs by 95%
- **Secure Execution**: Sandboxed environment for AI-generated code
- **Intelligent Fallbacks**: Local processing when AI services are unavailable

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | One of the AI keys |
| `GOOGLE_API_KEY` | Google Gemini API key | One of the AI keys |

### Advanced Settings

You can customize the behavior by modifying the configuration in `app.py`:

- **Sample Size**: Adjust the data sample size for AI processing
- **Chart Types**: Enable/disable specific visualization types
- **Security Settings**: Modify code execution restrictions

## 🚨 Troubleshooting

### Common Issues

**"No AI provider available"**
- Check your API keys are correctly set
- Verify you have billing enabled (for OpenAI)
- Try the alternative AI provider

**"Module not found" errors**
```bash
pip install --upgrade -r requirements.txt
```

**Charts not displaying**
- Check browser console for JavaScript errors
- Try refreshing the page
- Verify your data has appropriate columns for the requested chart

**Performance issues with large files**
- The app automatically optimizes for large datasets
- For files >100MB, consider preprocessing or sampling

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Powered by [LangChain](https://langchain.com/) for AI orchestration
- Visualizations by [Plotly](https://plotly.com/)
- Data processing with [Pandas](https://pandas.pydata.org/)

## 📞 Support

- 🐛 **Bug Reports**: [Open an issue](https://github.com/your-username/datacraft-supertool/issues)
- 💡 **Feature Requests**: [Start a discussion](https://github.com/your-username/datacraft-supertool/discussions)
- 📧 **Questions**: Create an issue with the "question" label

---

<div align="center">

**Made with ❤️ for the data community**

*Transform your data analysis journey with the power of conversational AI*

[⭐ Star this repo](https://github.com/your-username/datacraft-supertool) • [🍴 Fork it](https://github.com/your-username/datacraft-supertool/fork) • [📄 Documentation](DETAILS.md)

</div>