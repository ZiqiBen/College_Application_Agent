# College Application Helper - AI-Powered Document Generator

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)](https://python.langchain.com/)

An intelligent system for generating high-quality graduate school application documents (Personal Statements, Resumes, Recommendation Letters) using advanced AI techniques including RAG, ReAct, Reflection, and Reflexion.

## 🎯 Features

### ✨ Writing Agent v2.0 (NEW!)

The latest generation system powered by LangGraph and LangChain:

- **🧠 Intelligent Generation**: Deep content understanding using GPT-4/Claude, not template-based
- **🔍 RAG (Retrieval-Augmented Generation)**: Retrieves relevant program info and applicant experiences
- **🛠️ ReAct (Reasoning + Acting)**: Tool-augmented reasoning with 4 specialized tools
- **🎯 Reflection**: Multi-dimensional quality evaluation (5 dimensions)
- **🔄 Reflexion**: Memory-enhanced learning from past iterations
- **📋 ReWOO**: Strategic planning before generation

### 📄 Document Types

1. **Personal Statement (PS)**: 500-800 words, personalized narrative
2. **Resume Bullets**: Quantified achievements with strong action verbs
3. **Recommendation Letter**: Professional endorsement with concrete examples

### 🔧 Multiple Generation Systems

- **Writing Agent v2.0** (Recommended): LangGraph-based, highest quality
- **Multi-Agent System**: Writer-Critic iterative improvement
- **Simple Generator**: Fast template-based fallback

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
  - [API Usage](#api-usage)
  - [Web Interface](#web-interface)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Quality Evaluation](#-quality-evaluation)
- [Advanced Features](#-advanced-features)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## 🚀 Quick Start

### Prerequisites

- Python >= 3.9.6
- OpenAI API key (or Anthropic/Qwen)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```env
OPENAI_API_KEY=sk-your-key-here
WRITING_AGENT_LLM_PROVIDER=openai
```

### 3. Start the Server

```bash
python -m src.rag_service.api
```

The API will be available at `http://localhost:8000`

### 4. (Optional) Start Web Interface

```bash
streamlit run streamlit_app.py
```

Access the web app at `http://localhost:8501`

### 5. Test the API

Visit `http://localhost:8000/docs` for interactive API documentation.

## 📦 Installation

### System Requirements

- **Python**: 3.9.6 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for dependencies and models

### Detailed Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd College_Application_Agent
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Key Dependencies

- **Core**: FastAPI, Streamlit, LangChain, LangGraph
- **LLM Integration**: langchain-openai, langchain-anthropic, openai, anthropic
- **ML/NLP**: sentence-transformers, torch, scikit-learn
- **Vector Storage**: faiss-cpu, chromadb
- **Utilities**: pydantic, python-dotenv, beautifulsoup4

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following:

```env
# LLM Provider (openai, anthropic, or qwen)
WRITING_AGENT_LLM_PROVIDER=openai

# OpenAI (Recommended)
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4-turbo-preview

# Anthropic (Optional)
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Qwen (Optional)
QWEN_API_KEY=your_key_here
QWEN_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Generation Settings
MAX_ITERATIONS=3
QUALITY_THRESHOLD=0.85
WRITING_AGENT_TEMPERATURE=0.7
RETRIEVAL_TOP_K=5
```

### Configuration Presets

**For Important Applications** (PhD, Top Programs):
```env
MAX_ITERATIONS=5
QUALITY_THRESHOLD=0.90
OPENAI_MODEL=gpt-4
```

**For Regular Applications**:
```env
MAX_ITERATIONS=3
QUALITY_THRESHOLD=0.85
OPENAI_MODEL=gpt-4-turbo-preview
```

**For Quick Drafts**:
```env
MAX_ITERATIONS=2
QUALITY_THRESHOLD=0.75
OPENAI_MODEL=gpt-3.5-turbo
```

## 💻 Usage

### API Usage

#### Example: Generate Personal Statement

```python
import requests

url = "http://localhost:8000/generate/writing-agent"

data = {
    "profile": {
        "name": "Alice Zhang",
        "major": "Data Science",
        "gpa": 3.78,
        "skills": ["Python", "Machine Learning", "SQL"],
        "experiences": [
            {
                "title": "Data Analyst Intern",
                "org": "TechCorp",
                "impact": "Built ML model improving accuracy by 15%",
                "skills": ["Python", "TensorFlow"]
            }
        ],
        "goals": "Apply ML to real-world analytics and product data science"
    },
    "program_text": "Columbia University MS in Data Science...",
    "document_type": "personal_statement",
    "llm_provider": "openai",
    "max_iterations": 3,
    "quality_threshold": 0.85
}

response = requests.post(url, json=data)
result = response.json()

print("Generated Document:")
print(result["document"])
print("\nQuality Score:", result["quality_report"]["final_score"])
```

#### API Endpoints

- `POST /generate/writing-agent` - **NEW**: LangGraph-based generation (recommended)
- `POST /generate` - Multi-agent or simple generation with auto-selection
- `POST /generate/multi-agent` - Force multi-agent system
- `POST /generate/simple` - Force simple template system
- `GET /health` - Health check
- `GET /systems/info` - System information
- `GET /docs` - Interactive API documentation

### Web Interface

1. Start Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Open browser to `http://localhost:8501`

3. Fill in your profile information

4. Select document type and system

5. Click "Generate" and wait for results

## 🏗️ System Architecture

### Writing Agent v2.0 Workflow

```
┌────────────────────────────────────────────────────────────┐
│                  LangGraph State Machine                    │
└────────────────────────────────────────────────────────────┘
                           │
       ┌───────────────────┼──────────────────┐
       ▼                   ▼                  ▼
  ┌─────────┐        ┌──────────┐      ┌──────────┐
  │  Plan   │───────▶│   RAG    │─────▶│  ReAct   │
  │  Node   │        │  Node    │      │  Node    │
  └─────────┘        └──────────┘      └──────────┘
                                             │
                                             ▼
                                       ┌──────────┐
                                       │ Reflect  │
                                       │  Node    │
                                       └──────────┘
                                             │
                    ┌────────────────────────┴────────┐
                    ▼                                 ▼
             ┌──────────┐                      ┌──────────┐
             │  Revise  │                      │ Finalize │
             │  Node    │                      │          │
             └──────────┘                      └──────────┘
                    │                                 │
                    └────────► Loop ◄────────────────┘
```

### Node Functions

1. **Plan Node**: Analyzes task and creates generation strategy (ReWOO)
2. **RAG Node**: Retrieves relevant program info and applicant experiences
3. **ReAct Node**: Uses tools to gather info and generate initial draft
4. **Reflect Node**: Evaluates quality across 5 dimensions
5. **Revise Node**: Improves content based on reflection feedback
6. **Finalize Node**: Prepares final output and updates memory

### Tools (ReAct)

- **Match Calculator**: Calculates profile-program alignment score
- **Keyword Extractor**: Extracts program-relevant keywords
- **Experience Finder**: Finds most relevant experiences from profile
- **Requirement Checker**: Extracts program-specific requirements

## 📁 Project Structure

```
College_Application_Agent/
├── src/
│   ├── rag_service/              # Original RAG-based service
│   │   ├── api.py                # FastAPI endpoints
│   │   ├── generator.py          # Simple generator (legacy)
│   │   ├── multi_agent_generator.py  # Multi-agent system
│   │   ├── retriever_bert.py     # BERT-based retrieval
│   │   └── ingest.py             # Corpus ingestion
│   │
│   └── writing_agent/            # NEW: LangGraph-based system
│       ├── __init__.py
│       ├── config.py             # Configuration management
│       ├── state.py              # State definitions
│       ├── graph.py              # LangGraph workflow
│       ├── memory.py             # Reflexion memory
│       ├── llm_utils.py          # LLM utilities
│       ├── nodes/                # Workflow nodes
│       │   ├── plan_node.py
│       │   ├── rag_node.py
│       │   ├── react_node.py
│       │   ├── reflect_node.py
│       │   └── revise_node.py
│       ├── tools/                # ReAct tools
│       │   ├── match_calculator.py
│       │   ├── keyword_extractor.py
│       │   ├── experience_finder.py
│       │   └── requirement_checker.py
│       └── prompts/              # Prompt templates
│           ├── ps_prompts.py
│           ├── resume_prompts.py
│           ├── rl_prompts.py
│           └── reflection_prompts.py
│
├── data/
│   ├── corpus/                   # Program data (JSON files)
│   └── samples/                  # Sample profiles
│       ├── profile_sample.json
│       └── resume_sample.txt
│
├── data_preparation/             # Data scraping & processing
│   ├── pipeline_Version2.py
│   ├── schema_Version2.py
│   └── README_data_preparation.md
│
├── streamlit_app.py              # Web interface
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── README.md                     # This file
├── QUICKSTART.md                 # Quick start guide
└── out/                          # Output directory (generated)
    └── writing_agent_cache/      # Memory cache
```

## 📊 Quality Evaluation

### Evaluation Dimensions

The Writing Agent v2.0 evaluates documents across 5 dimensions:

1. **Keyword Coverage (20%)**: Are required keywords naturally integrated?
2. **Personalization (25%)**: Is content specific with concrete examples?
3. **Coherence (20%)**: Is structure logical and well-organized?
4. **Program Alignment (20%)**: Does it connect to specific program features?
5. **Persuasiveness (15%)**: Is it compelling and convincing?

### Scoring System

- **Score Range**: 0.0 - 1.0 for each dimension
- **Overall Score**: Weighted average of all dimensions
- **Approval Threshold**: Default 0.85 (configurable)
- **Iterations**: Continue until approved or max iterations reached

### Example Quality Report

```json
{
  "final_score": 0.89,
  "total_iterations": 2,
  "iteration_history": [...],
  "dimension_scores": {
    "keyword_coverage": 0.85,
    "personalization": 0.92,
    "coherence": 0.88,
    "program_alignment": 0.90,
    "persuasiveness": 0.87
  },
  "approved": true
}
```

## 🎓 Advanced Features

### Custom Corpus

Prepare your own program corpus for better RAG:

1. Create JSON files in `data/corpus/`
2. Follow the schema in `data_preparation/schema_Version2.py`
3. Use the data preparation pipeline for scraping

### Memory & Learning

The system learns from past generations:

- Tracks successful patterns
- Records common issues and resolutions
- Suggests strategies based on history
- Persists to disk: `out/writing_agent_cache/reflexion_memory.json`

### Customization

1. **Modify Prompts**: Edit files in `src/writing_agent/prompts/`
2. **Add Tools**: Create new tools in `src/writing_agent/tools/`
3. **Adjust Workflow**: Modify `src/writing_agent/graph.py`
4. **Change Evaluation**: Edit `src/writing_agent/prompts/reflection_prompts.py`

## 🔧 Troubleshooting

### Common Issues

**Issue**: ImportError for langchain/langgraph
```bash
pip install langchain langgraph langchain-openai langchain-anthropic
```

**Issue**: API key not found
```bash
# Check .env file exists
cat .env  # or type .env on Windows

# Verify API key is loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"
```

**Issue**: Low quality scores
- Increase `max_iterations` (e.g., 5)
- Use stronger model (e.g., gpt-4)
- Provide more detailed profile information
- Lower `quality_threshold` (e.g., 0.75)

**Issue**: Generation is slow
- Normal for LLM-based systems (10-20 seconds)
- Use faster model (gpt-3.5-turbo) for drafts
- Reduce `max_iterations`

### Debug Mode

Enable detailed logging:

```env
LOG_LEVEL=DEBUG
```

Check logs in `logs/` directory (if configured).

## 📈 Performance Comparison

| Feature | Simple | Multi-Agent | Writing Agent v2.0 |
|---------|--------|-------------|-------------------|
| Generation Method | Templates | Iterative | LLM + Workflow |
| Quality | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Personalization | Low | Medium | High |
| Intelligence | Rule-based | Critic feedback | LLM reasoning |
| Speed | <1s | 3-10s | 10-20s |
| Cost | Free | Low | Medium |
| Flexibility | Poor | Good | Excellent |

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

See LICENSE file for details.

## 📞 Support

- **Documentation**: Check `QUICKSTART.md` and `src/writing_agent/README.md`
- **Issues**: Open an issue on GitHub
- **API Docs**: Visit `/docs` endpoint when server is running

## 🙏 Acknowledgments

- Built with [LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- Powered by OpenAI GPT-4, Anthropic Claude, or Qwen models
- Uses BERT-based embeddings via sentence-transformers

## 📚 Additional Resources

- [Quick Start Guide](QUICKSTART.md) - Get started in minutes
- [Writing Agent Documentation](src/writing_agent/README.md) - Detailed technical docs
- [Data Preparation Guide](data_preparation/README_data_preparation.md) - Corpus creation

---

**Version**: 2.0  
**Last Updated**: 2025-11-16  
**Status**: Production Ready
