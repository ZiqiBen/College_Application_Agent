# College Application Helper V4.0

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)](https://python.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-red.svg)](https://streamlit.io/)

An intelligent end-to-end system for graduate school applications. The system combines **Smart Program Matching** with **AI-Powered Document Generation** to help applicants find the best-fit programs and create high-quality application documents (Personal Statements, Resume Bullets, Recommendation Letters).

## 🎯 Key Features

### 🆕 V4.0 Highlights

- **Dual Dataset Support**: Choose between V1 (legacy) and V2 (enhanced) datasets
- **6-Dimension Matching** (V2): Academic, Skills, Experience, Goals, Requirements, + **Curriculum Alignment**
- **LLM-Powered Fit Reasons**: Personalized explanations for why each program fits you
- **Course-Level Analysis**: Match your skills against specific program courses (V2)
- **Complete Workflow**: Profile Input → Smart Matching → Program Selection → Document Generation

### 🔍 Smart Program Matching Service

Intelligently matches your profile against 80+ graduate programs from top universities:

| Dimension | What It Evaluates | Weight |
|-----------|-------------------|--------|
| **Academic** | GPA, major relevance, coursework alignment | 25% |
| **Skills** | Technical skills coverage and depth | 20% |
| **Experience** | Work experience relevance and impact | 15% |
| **Goals** | Career goals alignment with program mission | 20% |
| **Requirements** | Application requirements compliance | 10% |
| **Curriculum** (V2) | Course-level skill matching | 10% |

### ✍️ Writing Agent (LangGraph-Based)

Advanced document generation powered by LangGraph with multiple AI techniques:

- **📋 ReWOO Planning**: Strategic plan before generation based on document type
- **🔍 RAG**: Retrieves relevant program information and applicant experiences
- **🛠️ ReAct**: Tool-augmented reasoning with 4 specialized tools
- **🎯 Reflection**: Multi-dimensional quality evaluation (5 dimensions)
- **🔄 Reflexion**: Memory-enhanced learning from past iterations
- **📈 Adaptive Weights**: Dynamically adjusts focus based on weakest dimensions

### 📄 Document Types

| Document | Description | Target Length |
|----------|-------------|---------------|
| **Personal Statement** | Personalized narrative connecting your background to program | 500-800 words |
| **Resume Bullets** | Quantified achievements with strong action verbs | 1-2 bullets per experience |
| **Recommendation Letter** | Professional endorsement template with concrete examples | 400-600 words |

### 🔧 Multiple Generation Systems

| System | Quality | Speed | Best For |
|--------|---------|-------|----------|
| **Writing Agent** | ⭐⭐⭐⭐⭐ | 10-20s | Important applications |
| **Multi-Agent** | ⭐⭐⭐⭐ | 3-10s | Balanced quality/speed |
| **Simple Generator** | ⭐⭐⭐ | <1s | Quick drafts |

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
  - [Web Interface (Recommended)](#web-interface-recommended)
  - [API Usage](#api-usage)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Datasets](#-datasets)
- [Matching Service Details](#-matching-service-details)
- [Writing Agent Details](#-writing-agent-details)
- [Quality Evaluation](#-quality-evaluation)
- [Data Preparation](#-data-preparation)
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

Create a `.env` file:

```env
OPENAI_API_KEY=sk-your-key-here
WRITING_AGENT_LLM_PROVIDER=openai
V2_LLM_PROVIDER=openai
```

### 3. Start the Backend Server

```bash
python -m src.rag_service.api
```

The API will be available at `http://localhost:8000`

### 4. Start the Web Interface

```bash
streamlit run streamlit_app.py
```

Access the web app at `http://localhost:8501`

### 5. Use the Application

1. **Enter your profile** (major, GPA, skills, experiences, goals)
2. **Select dataset** (V2 Enhanced recommended)
3. **Click "Start Matching"** to find best-fit programs
4. **Select a program** from the ranked results
5. **Generate documents** (PS, Resume, Recommendation Letter)

## 📦 Installation

### System Requirements

- **Python**: 3.9.6 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for dependencies and models

### Detailed Setup

```bash
# Clone the repository
git clone <repository-url>
cd College_Application_Agent

# Create virtual environment (recommended)
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

| Category | Packages |
|----------|----------|
| **Web Framework** | FastAPI, Streamlit, uvicorn |
| **AI/LLM** | LangChain, LangGraph, langchain-openai, langchain-anthropic |
| **Embeddings** | sentence-transformers, torch |
| **Vector Store** | faiss-cpu, chromadb |
| **Utilities** | pydantic, python-dotenv, beautifulsoup4, requests |

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with your settings:

```env
# LLM Provider for Writing Agent (openai, anthropic, or qwen)
WRITING_AGENT_LLM_PROVIDER=openai

# LLM Provider for V2 Matching Explainer
V2_LLM_PROVIDER=openai

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

**For Quick Drafts**:
```env
MAX_ITERATIONS=2
QUALITY_THRESHOLD=0.75
OPENAI_MODEL=gpt-3.5-turbo
```

## 💻 Usage

### Web Interface (Recommended)

The Streamlit frontend provides a complete workflow with three main pages:

#### Page 1: Smart Matching + Generation (Main Workflow)

Complete end-to-end flow:
1. **Input Profile**: Enter your name, major, GPA, skills, courses, experiences, and goals
2. **Configure Matching**: Select dataset (V1/V2), set top K programs, minimum score, dimension weights
3. **View Results**: See ranked programs with dimension scores, fit reasons, and course matches
4. **Select Program**: Choose your target program from the matches
5. **Generate Documents**: Create Personal Statement, Resume Bullets, or Recommendation Letter

#### Page 2: Quick Generation (Manual Input)

For users who already know their target program:
- Enter program information manually
- Generate documents without matching step

#### Page 3: Dashboard

View generation history and results.

### API Usage

#### Start the API Server

```bash
python -m src.rag_service.api
```

#### Main Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and available endpoints |
| `/health` | GET | Health check |
| `/generate/writing-agent` | POST | LangGraph-based document generation |
| `/generate` | POST | Multi-agent or simple generation |
| `/match/programs` | POST | V1 program matching |
| `/v2/match/programs` | POST | V2 enhanced program matching |
| `/v2/match/program/{id}/details` | GET | Get V2 program details |
| `/v2/match/programs/list` | GET | List all V2 programs |
| `/docs` | GET | Interactive API documentation (Swagger) |

#### Example: Match Programs (V2)

```python
import requests

url = "http://localhost:8000/v2/match/programs"

payload = {
    "profile": {
        "name": "Alice Zhang",
        "major": "Data Science",
        "gpa": 3.78,
        "skills": ["Python", "Machine Learning", "SQL", "PyTorch"],
        "courses": ["Machine Learning", "Deep Learning", "Statistical Inference"],
        "experiences": [
            {
                "title": "Data Science Intern",
                "org": "Tech Company",
                "impact": "Built ML model improving prediction accuracy by 15%",
                "skills": ["Python", "TensorFlow"]
            }
        ],
        "goals": "Apply ML to real-world analytics and product data science"
    },
    "top_k": 10,
    "min_score": 0.5,
    "include_curriculum_analysis": True
}

response = requests.post(url, json=payload)
matches = response.json()["matches"]

for match in matches[:3]:
    print(f"{match['university']} - {match['program_name']}")
    print(f"  Score: {match['overall_score']:.2f}")
    print(f"  Fit Reasons: {match['fit_reasons'][:2]}")
```

#### Example: Generate Personal Statement

```python
import requests

url = "http://localhost:8000/generate/writing-agent"

payload = {
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
    "resume_text": "Your current resume text...",
    "program_text": "Columbia University MS in Data Science program description...",
    "document_type": "personal_statement",
    "llm_provider": "openai",
    "max_iterations": 3,
    "quality_threshold": 0.85
}

response = requests.post(url, json=payload)
result = response.json()

print("Generated Document:")
print(result["document"])
print(f"\nQuality Score: {result['quality_report']['final_score']:.2f}")
print(f"Iterations: {result['iterations']}")
```

## 🏗️ System Architecture

### Overall System Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Streamlit Frontend (V4.0)                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Profile   │───▶│   Smart     │───▶│   Select    │───▶│  Generate   │  │
│  │   Input     │    │   Matching  │    │   Program   │    │  Documents  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌───────────────────────┐       ┌───────────────────────┐
        │   Matching Service    │       │    Writing Agent      │
        │  ┌─────────────────┐  │       │  ┌─────────────────┐  │
        │  │   V1 Matcher    │  │       │  │   LangGraph     │  │
        │  │  (Legacy Data)  │  │       │  │   Workflow      │  │
        │  └─────────────────┘  │       │  └─────────────────┘  │
        │  ┌─────────────────┐  │       │  ┌─────────────────┐  │
        │  │   V2 Matcher    │  │       │  │   Multi-Agent   │  │
        │  │ (Enhanced Data) │  │       │  │   Generator     │  │
        │  └─────────────────┘  │       │  └─────────────────┘  │
        └───────────────────────┘       └───────────────────────┘
```

### Writing Agent LangGraph Workflow

```
┌────────────────────────────────────────────────────────────┐
│                  LangGraph State Machine                    │
└────────────────────────────────────────────────────────────┘
                           │
       ┌───────────────────┼──────────────────┐
       ▼                   ▼                  ▼
  ┌─────────┐        ┌──────────┐      ┌──────────┐
  │  Plan   │───────▶│   RAG    │─────▶│  ReAct   │
  │ (ReWOO) │        │  Node    │      │ Generate │
  └─────────┘        └──────────┘      └──────────┘
       │                                     │
       │ Strategy                            │ Initial Draft
       │                                     ▼
       │                              ┌──────────┐
       │                              │ Reflect  │◄────────┐
       │                              │  Node    │         │
       │                              └──────────┘         │
       │                                     │             │
       │                    ┌────────────────┴────────┐    │
       │                    ▼                         ▼    │
       │             ┌──────────┐              ┌──────────┐│
       │             │  Revise  │─────────────▶│ Finalize ││
       │             │  Node    │  (if done)   │          ││
       │             └──────────┘              └──────────┘│
       │                    │                              │
       │                    └────── Loop (if needed) ─────┘
       └──────────────────────────────────────────────────────
```

### Matching Service V2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ProgramMatcherV2                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  DimensionScorer │  │  MatchExplainer │  │   Models    │ │
│  │       V2        │  │       V2        │  │     V2      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│           │                    │                  │         │
│  ┌────────┴────────────────────┴──────────────────┘         │
│  │                                                          │
│  │  6 Dimensions:                                           │
│  │  ├── Academic (GPA, Major, Courses)                     │
│  │  ├── Skills (Technical skills coverage)                 │
│  │  ├── Experience (Work/Project relevance)                │
│  │  ├── Goals (Career alignment)                           │
│  │  ├── Requirements (Application compliance)              │
│  │  └── Curriculum (V2: Course-level matching)             │
│  │                                                          │
│  │  Features:                                               │
│  │  ├── Semantic similarity (sentence-transformers)        │
│  │  ├── LLM-powered fit reasons generation                 │
│  │  ├── Batch processing for efficiency                    │
│  │  └── Course description analysis                        │
│  │                                                          │
└──┴──────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
College_Application_Agent/
├── src/
│   ├── matching_service/           # Program Matching Service
│   │   ├── __init__.py
│   │   ├── matcher.py              # V1 Matcher (legacy)
│   │   ├── matcher_v2.py           # V2 Matcher (enhanced dataset)
│   │   ├── models.py               # V1 Data models
│   │   ├── models_v2.py            # V2 Data models (6 dimensions)
│   │   ├── scorer.py               # V1 Scoring algorithms
│   │   ├── scorer_v2.py            # V2 Scoring (with curriculum)
│   │   ├── explainer.py            # V1 Explanation generator
│   │   ├── explainer_v2.py         # V2 LLM-powered explainer
│   │   └── README.md               # Matching service docs
│   │
│   ├── rag_service/                # RAG & API Service
│   │   ├── api.py                  # FastAPI endpoints (V4.0)
│   │   ├── generator.py            # Simple generator (legacy)
│   │   ├── multi_agent_generator.py # Multi-agent system
│   │   ├── retriever_bert.py       # BERT-based retrieval
│   │   └── ingest.py               # Corpus ingestion
│   │
│   └── writing_agent/              # LangGraph Writing Agent
│       ├── __init__.py
│       ├── config.py               # Configuration management
│       ├── state.py                # WritingState definitions
│       ├── graph.py                # LangGraph workflow
│       ├── memory.py               # Reflexion memory system
│       ├── llm_utils.py            # LLM provider utilities
│       ├── nodes/                  # Workflow nodes
│       │   ├── plan_node.py        # ReWOO planning
│       │   ├── rag_node.py         # RAG retrieval
│       │   ├── react_node.py       # ReAct generation
│       │   ├── reflect_node.py     # Self-evaluation
│       │   └── revise_node.py      # Improvement
│       ├── tools/                  # ReAct tools
│       │   ├── match_calculator.py
│       │   ├── keyword_extractor.py
│       │   ├── experience_finder.py
│       │   └── requirement_checker.py
│       ├── prompts/                # Prompt templates
│       │   ├── ps_prompts.py       # Personal Statement
│       │   ├── resume_prompts.py   # Resume Bullets
│       │   ├── rl_prompts.py       # Recommendation Letter
│       │   └── reflection_prompts.py # Evaluation
│       └── README.md               # Writing agent docs
│
├── data/
│   ├── corpus/                     # V1 Legacy corpus (80+ programs)
│   │   ├── Harvard-*.json
│   │   ├── Stanford-*.json
│   │   ├── MIT-*.json
│   │   └── ... (more universities)
│   └── samples/                    # Sample profiles
│       ├── profile_sample.json
│       └── resume_sample.txt
│
├── data_preparation/               # Data scraping & processing
│   ├── pipeline_Version2.py        # V2 data pipeline
│   ├── schema_Version2.py          # V2 JSON schema
│   ├── qwen_adapter_Version2.py    # Qwen LLM adapter
│   ├── gpt_adapter.py              # GPT adapter
│   ├── utils_Version2.py           # Utilities
│   └── README_data_preparation.md  # Data prep docs
│
├── streamlit_app.py                # Web interface (V4.0)
├── streamlit_app_old.py            # Legacy web interface
├── requirements.txt                # Python dependencies
├── QUICKSTART.md                   # Quick start guide
└── README.md                       # This file
```

## 📊 Datasets

### V1 Legacy Dataset (`data/corpus/`)

- **Structure**: Flat JSON files with sections-based organization
- **Programs**: 80+ graduate programs from top universities
- **Fields**: name, university, field, description_text, core_courses, focus_areas, required_skills
- **Matching**: 5 dimensions (academic, skills, experience, goals, requirements)

### V2 Enhanced Dataset (`data_preparation/dataset/`)

- **Structure**: Nested directories by domain (e.g., `seas.harvard.edu/...`)
- **Schema**: ExtractedFields with rich nested structures
- **Features**:
  - Courses with **descriptions** (not just names)
  - Structured application requirements
  - Program background (mission, faculty, resources)
  - Training outcomes (career paths, research orientation)
- **Matching**: 6 dimensions (+ curriculum alignment)

### V2 Schema Overview

```python
class ExtractedFields:
    program_name: str
    school: str
    department: str
    duration: str
    courses: List[Course]  # name + description
    application_requirements: ApplicationRequirements
    program_background: ProgramBackground
    training_outcomes: TrainingOutcomes
    tuition: str
    contact_email: str
```

## 🔍 Matching Service Details

### V2 Dimension Scoring

| Dimension | Factors | Weight |
|-----------|---------|--------|
| **Academic** | GPA match, Major relevance, Coursework alignment | 25% |
| **Skills** | Required skills coverage, Course-extracted skills, Skill breadth | 20% |
| **Experience** | Experience count, Relevance to program, Impact level | 15% |
| **Goals** | Career goal alignment, Program mission match | 20% |
| **Requirements** | GPA requirement, Tests, Prerequisites | 10% |
| **Curriculum** | Course-level skill matching, Curriculum fit | 10% |

### LLM-Powered Fit Reasons

V2 uses LLM to generate personalized "Why This Program Fits You" reasons:

```python
# Example fit reasons
[
    "Your Machine Learning coursework directly aligns with the program's ML focus areas",
    "Your data science internship experience matches the program's industry-oriented curriculum",
    "Your career goals in product analytics align with the program's professional orientation"
]
```

### Match Levels

| Level | Score Range | Description |
|-------|-------------|-------------|
| 🟢 Excellent | 0.85+ | Strong match across all dimensions |
| 🟡 Good | 0.70-0.85 | Good overall fit with minor gaps |
| 🟠 Moderate | 0.55-0.70 | Reasonable fit, some areas to strengthen |
| 🟤 Fair | 0.40-0.55 | Partial match, significant preparation needed |
| 🔴 Weak | <0.40 | Limited alignment |

## ✍️ Writing Agent Details

### Nodes Description

| Node | Function | Key Features |
|------|----------|--------------|
| **Plan** | Creates generation strategy | Document-type-specific planning (ReWOO) |
| **RAG** | Retrieves relevant context | Program info, applicant experiences |
| **Generate** | Creates initial draft | ReAct with 4 tools |
| **Reflect** | Evaluates quality | 5 dimensions, adaptive weights |
| **Revise** | Improves draft | Targeted fixes based on weakest dimensions |
| **Finalize** | Outputs result | Memory update, quality report |

### ReAct Tools

1. **Match Calculator**: Computes profile-program alignment score
2. **Keyword Extractor**: Extracts must-have keywords for document
3. **Experience Finder**: Identifies most relevant experiences
4. **Requirement Checker**: Extracts program-specific requirements

### Reflexion Memory

The system learns from past generations:

- Records successful patterns and strategies
- Tracks dimension-level improvements across iterations
- Suggests strategies based on history
- Persists to: `out/writing_agent_cache/reflexion_memory.json`

## 📊 Quality Evaluation

### Evaluation Dimensions (Reflection Node)

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| **Keyword Coverage** | 20% | Integration of required keywords |
| **Personalization** | 25% | Specificity with concrete examples |
| **Coherence** | 20% | Logical structure and flow |
| **Program Alignment** | 20% | Connection to program features |
| **Persuasiveness** | 15% | Compelling and convincing tone |

### Adaptive Weighting

The system dynamically adjusts dimension weights based on:
- Current overall quality score
- Weakest dimensions in previous iteration
- Stagnation detection (minimal improvement)

### Quality Report Example

```json
{
  "final_score": 0.89,
  "quality_threshold": 0.85,
  "total_iterations": 2,
  "approved": true,
  "dimension_scores": {
    "keyword_coverage": 0.85,
    "personalization": 0.92,
    "coherence": 0.88,
    "program_alignment": 0.90,
    "persuasiveness": 0.87
  },
  "dimension_improvements": {
    "keyword_coverage": 0.12,
    "personalization": 0.08
  },
  "score_improvement": 0.15
}
```

## 🛠️ Data Preparation

### Creating New Program Data

Use the V2 pipeline to scrape and process new program pages:

```bash
cd data_preparation
python pipeline_Version2.py --url "https://example.edu/program" --output ./dataset/
```

### V2 Pipeline Features

- **Web scraping** with BeautifulSoup
- **LLM extraction** using GPT or Qwen
- **Schema validation** with Pydantic
- **Chunking** for RAG embeddings

### Schema Reference

See `data_preparation/schema_Version2.py` for complete field definitions.

## 🔧 Troubleshooting

### Common Issues

**Issue**: Cannot connect to API server
```bash
# Ensure server is running
python -m src.rag_service.api

# Check if port 8000 is available
netstat -an | findstr 8000
```

**Issue**: ImportError for langchain/langgraph
```bash
pip install langchain langgraph langchain-openai langchain-anthropic
```

**Issue**: V2 Matcher not loading
```bash
# Ensure V2 dataset exists
ls data_preparation/dataset/graduate_programs/

# Check API initialization logs
python -m src.rag_service.api
# Look for: "✅ V2 Program Matcher initialized with X programs"
```

**Issue**: Low quality scores
- Increase `max_iterations` (e.g., 5)
- Use stronger model (e.g., gpt-4)
- Provide more detailed profile (experiences, specific goals)
- Lower `quality_threshold` for faster results

**Issue**: Matching timeout
- Reduce `top_k` (number of programs)
- Disable LLM fit reasons: `use_llm_explanation=False`
- Use V1 matcher (faster, less features)

### Debug Mode

```env
LOG_LEVEL=DEBUG
```

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution

- Additional university program data
- New matching dimensions
- UI/UX improvements
- Performance optimizations
- Documentation improvements

## 📄 License

See LICENSE file for details.

## 📞 Support

- **Documentation**: Check module-specific README files in `src/*/README.md`
- **API Docs**: Visit `http://localhost:8000/docs` when server is running
- **Issues**: Open an issue on GitHub

## 🙏 Acknowledgments

- Built with [LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- Embeddings via [Sentence-Transformers](https://www.sbert.net/)
- Powered by OpenAI GPT-4, Anthropic Claude, or Alibaba Qwen
- UI with [Streamlit](https://streamlit.io/) and [FastAPI](https://fastapi.tiangolo.com/)

---

**Version**: 4.0  
**Last Updated**: 2025-12-10  
**Status**: Production Ready

**Universities Covered**: Harvard, Stanford, MIT, Columbia, Yale, Princeton, Cornell, Brown, Dartmouth, Duke, Northwestern, UPenn, Caltech, Chicago, Johns Hopkins, Rice, Vanderbilt, and more.
