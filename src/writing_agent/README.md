# Writing Agent v2.0 - LangGraph Based System

## Overview

This is a complete redesign of the writing agent system using modern AI workflows:

- **RAG (Retrieval-Augmented Generation)**: Retrieves relevant program information and applicant experiences
- **ReAct (Reasoning + Acting)**: Uses tools to gather information and make intelligent decisions
- **Reflection**: Self-evaluates generated content across multiple dimensions
- **Reflexion**: Learns from past iterations with memory
- **ReWOO**: Plans generation strategy before execution

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph State Machine                   │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   ┌─────────┐        ┌──────────┐       ┌──────────┐
   │ Plan    │───────▶│   RAG    │──────▶│  ReAct   │
   │ Node    │        │  Node    │       │  Node    │
   └─────────┘        └──────────┘       └──────────┘
                                               │
                                               ▼
                                         ┌──────────┐
                                         │ Reflect  │
                                         │ Node     │
                                         └──────────┘
                                               │
                     ┌─────────────────────────┴────────┐
                     ▼                                  ▼
              ┌──────────┐                       ┌──────────┐
              │ Revise   │                       │ Finalize │
              │ Node     │                       │          │
              └──────────┘                       └──────────┘
                     │                                  │
                     └────────────► Loop ◄─────────────┘
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...
# or
QWEN_API_KEY=...
```

### 3. Run the API

```bash
python -m src.rag_service.api
```

The API will start on `http://localhost:8000`

## Usage

### API Endpoint

**POST** `/generate/writing-agent`

### Request Body

```json
{
  "profile": {
    "name": "Alice Zhang",
    "email": "alice@example.com",
    "major": "Data Science",
    "gpa": 3.78,
    "courses": ["Machine Learning", "Algorithms"],
    "skills": ["Python", "R", "SQL"],
    "experiences": [
      {
        "title": "Data Analyst Intern",
        "org": "TechCorp",
        "impact": "Built ML model improving accuracy by 15%",
        "skills": ["Python", "TensorFlow"]
      }
    ],
    "goals": "Apply ML to real-world analytics"
  },
  "program_text": "Master of Science in Data Science...",
  "document_type": "personal_statement",
  "llm_provider": "openai",
  "max_iterations": 3,
  "quality_threshold": 0.85
}
```

### Document Types

- `personal_statement`: Personal Statement (PS)
- `resume_bullets`: Resume bullet points
- `recommendation_letter`: Recommendation Letter

### Response

```json
{
  "success": true,
  "document": "Generated document text...",
  "quality_report": {
    "final_score": 0.89,
    "total_iterations": 2,
    "approved": true
  },
  "metadata": {
    "llm_provider": "openai",
    "model_name": "gpt-4-turbo-preview"
  },
  "generation_time_seconds": 8.5
}
```

## Quality Evaluation

The system evaluates documents across 5 dimensions:

1. **Keyword Coverage (20%)**: Are required keywords naturally integrated?
2. **Personalization (25%)**: Is content specific with concrete examples?
3. **Coherence (20%)**: Is structure logical and well-organized?
4. **Program Alignment (20%)**: Does it connect to specific program features?
5. **Persuasiveness (15%)**: Is it compelling and convincing?

**Overall Score Calculation**: Weighted average of dimension scores
**Approval Threshold**: Default 0.85 (configurable)

## Iteration Flow

1. **Plan**: Analyze task and create generation strategy
2. **RAG**: Retrieve relevant program info and experiences
3. **Generate**: Create initial draft using tools and context
4. **Reflect**: Self-evaluate with LLM across 5 dimensions
5. **Revise**: Improve based on feedback (if score < threshold)
6. **Repeat**: Steps 4-5 until approved or max iterations reached

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WRITING_AGENT_LLM_PROVIDER` | LLM provider | `openai` |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4-turbo-preview` |
| `MAX_ITERATIONS` | Max refinement iterations | `3` |
| `QUALITY_THRESHOLD` | Approval threshold | `0.85` |
| `RETRIEVAL_TOP_K` | Top K RAG chunks | `5` |
| `WRITING_AGENT_TEMPERATURE` | LLM temperature | `0.7` |

### In-Code Configuration

See `src/writing_agent/config.py` for all configuration options.

## Tools

The ReAct node uses these tools:

- **Match Calculator**: Calculates profile-program alignment score
- **Keyword Extractor**: Extracts program-relevant keywords
- **Experience Finder**: Finds most relevant experiences from profile
- **Requirement Checker**: Extracts program-specific requirements

## Memory & Learning

The system uses **Reflexion** to learn from past generations:

- Tracks successful patterns
- Records common issues and resolutions
- Suggests strategies based on history
- Persists memory to disk for future use

## Comparison with Old System

| Feature | Old System | New System |
|---------|-----------|------------|
| **Generation Method** | Template + filling | LLM-powered |
| **Refinement** | Keyword checks | Multi-dimensional evaluation |
| **Intelligence** | Rule-based | LLM reasoning |
| **Personalization** | Low | High |
| **Quality** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Speed** | Fast | Moderate |
| **Cost** | Free | API calls |

## Troubleshooting

### Import Errors

If you see import errors for `langchain` or `langgraph`:

```bash
pip install langchain langgraph langchain-openai langchain-anthropic
```

### API Key Errors

Ensure your `.env` file has the correct API key:

```bash
# Check if .env exists
ls -la .env

# Verify API key is set
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"
```

### Low Quality Scores

If generation quality is low:

1. Increase `max_iterations` (e.g., 5)
2. Lower `quality_threshold` (e.g., 0.75)
3. Provide more detailed profile information
4. Use more powerful model (e.g., `gpt-4` instead of `gpt-3.5-turbo`)

## Development

### Adding New Tools

Create a new tool in `src/writing_agent/tools/`:

```python
from langchain.tools import tool

@tool
def my_new_tool(arg1: str, arg2: int) -> dict:
    """Tool description"""
    # Implementation
    return {"result": "..."}
```

Register in `tools/__init__.py` and use in `react_node.py`.

### Adding New Nodes

Create a new node in `src/writing_agent/nodes/`:

```python
def my_new_node(state: WritingState) -> Dict[str, Any]:
    """Node implementation"""
    # Process state
    return {"updated_field": "value"}
```

Add to graph in `graph.py`.

## License

See main project LICENSE file.

## Support

For issues and questions, please open an issue on the GitHub repository.
