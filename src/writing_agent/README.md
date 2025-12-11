# ✍️ Writing Agent - LangGraph Based Document Generation

## Overview

The **Writing Agent** is an advanced AI-powered document generation system built on **LangGraph**. It generates high-quality application documents (Personal Statements, Resume Bullets, Recommendation Letters) using a sophisticated workflow that combines multiple AI techniques:

- **📋 ReWOO (Plan-Tool-Solve)**: Strategic planning before generation
- **🔍 RAG (Retrieval-Augmented Generation)**: Retrieves relevant program info and experiences
- **🛠️ ReAct (Reasoning + Acting)**: Tool-augmented reasoning with 4 specialized tools
- **🎯 Reflection**: Multi-dimensional self-evaluation (5 dimensions)
- **🔄 Reflexion**: Memory-enhanced learning from past iterations
- **📈 Adaptive Weights**: Dynamically adjusts focus based on weakest dimensions

---

## 🎯 Key Features

### **Document Types**

| Document | Description | Target Length |
|----------|-------------|---------------|
| **Personal Statement** | Personalized narrative connecting background to program | 500-800 words |
| **Resume Bullets** | Quantified achievements with strong action verbs | 1-2 bullets per experience |
| **Recommendation Letter** | Professional endorsement template with concrete examples | 400-600 words |

### **Quality Evaluation (5 Dimensions)**

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| **Keyword Coverage** | 20% | Are required keywords naturally integrated? |
| **Personalization** | 25% | Is content specific with concrete examples? |
| **Coherence** | 20% | Is structure logical and well-organized? |
| **Program Alignment** | 20% | Does it connect to specific program features? |
| **Persuasiveness** | 15% | Is it compelling and convincing? |

### **Generation Systems Comparison**

| Feature | Writing Agent | Multi-Agent | Simple Generator |
|---------|--------------|-------------|------------------|
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Speed** | 10-20s | 3-10s | <1s |
| **Intelligence** | LLM + Workflow | Critic feedback | Rule-based |
| **Personalization** | High | Medium | Low |
| **Cost** | Medium | Low | Free |

---

## 🏗️ Architecture

### **LangGraph State Machine**

```
                    ┌─────────────────────────────────────────────┐
                    │           LangGraph State Machine            │
                    └─────────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────────┐
                    │                 PLAN NODE                    │
                    │  • Analyzes document type                   │
                    │  • Creates generation strategy (ReWOO)      │
                    │  • Identifies key profile elements          │
                    └─────────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────────┐
                    │                  RAG NODE                    │
                    │  • Retrieves relevant program chunks        │
                    │  • Extracts program keywords                │
                    │  • Matches relevant experiences             │
                    └─────────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────────┐
                    │                REACT NODE                    │
                    │  • Calls 4 specialized tools                │
                    │  • Generates initial/revised draft          │
                    │  • Integrates tool results                  │
                    └─────────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────────┐
                    │               REFLECT NODE                   │
                    │  • 5-dimension quality evaluation           │
                    │  • Keyword integration analysis             │
                    │  • Identifies weakest dimensions            │
                    │  • Determines if revision needed            │
                    └─────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
            score < threshold                       score >= threshold
                    │                                       │
                    ▼                                       ▼
    ┌───────────────────────────┐       ┌───────────────────────────┐
    │        REVISE NODE        │       │       FINALIZE NODE       │
    │  • Targeted improvements  │       │  • Outputs final document │
    │  • Memory integration     │       │  • Creates quality report │
    │  • Dimension focus        │       │  • Updates Reflexion mem  │
    └───────────────────────────┘       └───────────────────────────┘
                    │
                    └──────────────► Loop back to REFLECT
```

### **Workflow Flow**

1. **Plan Node**: Creates document-specific generation strategy
2. **RAG Node**: Retrieves context from corpus and profile
3. **React Node**: Generates content using tools + LLM
4. **Reflect Node**: Evaluates quality across 5 dimensions
5. **Conditional**: If score < threshold → Revise, else → Finalize
6. **Revise Node**: Improves based on feedback (loops back to Reflect)
7. **Finalize Node**: Outputs document and updates memory

---

## 📂 Module Structure

```
src/writing_agent/
├── __init__.py              # Module exports
├── README.md                # This documentation
├── config.py                # Configuration management
├── state.py                 # WritingState TypedDict definitions
├── graph.py                 # LangGraph workflow definition
├── memory.py                # Reflexion memory system
├── llm_utils.py             # LLM provider utilities
│
├── nodes/                   # Workflow nodes
│   ├── __init__.py
│   ├── plan_node.py         # ReWOO planning
│   ├── rag_node.py          # RAG retrieval
│   ├── react_node.py        # ReAct generation
│   ├── reflect_node.py      # Self-evaluation
│   └── revise_node.py       # Content improvement
│
├── tools/                   # ReAct tools
│   ├── __init__.py
│   ├── match_calculator.py  # Profile-program match scoring
│   ├── keyword_extractor.py # Extract required keywords
│   ├── experience_finder.py # Find relevant experiences
│   └── requirement_checker.py # Extract program requirements
│
└── prompts/                 # Prompt templates
    ├── __init__.py
    ├── ps_prompts.py        # Personal Statement prompts
    ├── resume_prompts.py    # Resume Bullets prompts
    ├── rl_prompts.py        # Recommendation Letter prompts
    └── reflection_prompts.py # Evaluation prompts
```

---

## 📖 Usage

### **API Endpoint**

```
POST /generate/writing-agent
```

### **Request Body**

```json
{
    "profile": {
        "name": "Alice Zhang",
        "email": "alice@example.com",
        "major": "Data Science",
        "gpa": 3.78,
        "courses": ["Machine Learning", "Deep Learning", "Statistical Inference"],
        "skills": ["Python", "PyTorch", "SQL", "TensorFlow", "R"],
        "experiences": [
            {
                "title": "Data Science Intern",
                "org": "Tech Company",
                "impact": "Built ML model improving prediction accuracy by 15%",
                "skills": ["Python", "TensorFlow", "SQL"]
            },
            {
                "title": "Research Assistant",
                "org": "University AI Lab",
                "impact": "Analyzed healthcare datasets and created visualizations",
                "skills": ["Python", "R", "Data Visualization"]
            }
        ],
        "goals": "Apply ML to real-world analytics and product data science. Lead data-driven initiatives in tech companies."
    },
    "resume_text": "Your current resume content...",
    "program_text": "Columbia University MS in Data Science program description...",
    "document_type": "personal_statement",
    "llm_provider": "openai",
    "model_name": "gpt-4-turbo-preview",
    "temperature": 0.7,
    "max_iterations": 3,
    "quality_threshold": 0.85,
    "use_corpus": true,
    "retrieval_topk": 5
}
```

### **Response**

```json
{
    "success": true,
    "document": "Generated personal statement text...",
    "document_type": "personal_statement",
    "quality_report": {
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
        "score_improvement": 0.15,
        "keyword_analysis": {
            "required": ["Machine Learning", "Data Science", "Python"],
            "found": ["Machine Learning", "Data Science", "Python"],
            "overall_integration_quality": 0.85
        },
        "weakest_dimensions": ["keyword_coverage"]
    },
    "metadata": {
        "llm_provider": "openai",
        "model_name": "gpt-4-turbo-preview",
        "plan_created": true,
        "generation_completed": true,
        "revision_completed": true,
        "revision_count": 1
    },
    "generation_time_seconds": 12.5,
    "iterations": 2,
    "draft_history_length": 2
}
```

---

## 🔧 Configuration

### **Environment Variables**

| Variable | Description | Default |
|----------|-------------|---------|
| `WRITING_AGENT_LLM_PROVIDER` | LLM provider (openai/anthropic/qwen) | `openai` |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4-turbo-preview` |
| `ANTHROPIC_API_KEY` | Anthropic API key | Optional |
| `ANTHROPIC_MODEL` | Anthropic model name | `claude-3-sonnet-20240229` |
| `QWEN_API_KEY` | Qwen API key | Optional |
| `MAX_ITERATIONS` | Max refinement iterations | `3` |
| `QUALITY_THRESHOLD` | Approval threshold (0-1) | `0.85` |
| `RETRIEVAL_TOP_K` | RAG retrieval count | `5` |
| `WRITING_AGENT_TEMPERATURE` | LLM temperature | `0.7` |

### **Configuration Presets**

**Important Applications (PhD, Top Programs):**
```env
MAX_ITERATIONS=5
QUALITY_THRESHOLD=0.90
OPENAI_MODEL=gpt-4
```

**Quick Drafts:**
```env
MAX_ITERATIONS=2
QUALITY_THRESHOLD=0.75
OPENAI_MODEL=gpt-3.5-turbo
```

### **In-Code Configuration**

See `config.py` for all options:

```python
class WritingAgentConfig:
    # LLM settings
    TEMPERATURE = 0.7
    TEMPERATURE_REFLECTION = 0.3  # Lower for consistent evaluation
    
    # Quality settings
    MAX_ITERATIONS = 3
    QUALITY_THRESHOLD = 0.85
    
    # RAG settings
    RETRIEVAL_TOP_K = 5
    
    # Cache directory
    CACHE_DIR = "out/writing_agent_cache"
```

---

## 🛠️ Tools (ReAct)

The React node uses 4 specialized tools:

### **1. Match Calculator** (`match_calculator.py`)

Calculates profile-program alignment score.

```python
# Input
profile: Dict[str, Any]
program_info: Dict[str, Any]
program_keywords: List[str]

# Output
{
    "overall_score": 0.85,
    "skill_match": 0.90,
    "experience_relevance": 0.80,
    "goals_alignment": 0.82
}
```

### **2. Keyword Extractor** (`keyword_extractor.py`)

Extracts must-have keywords from program text.

```python
# Input
program_text: str

# Output
["Machine Learning", "Data Science", "Python", "Statistics", ...]
```

### **3. Experience Finder** (`experience_finder.py`)

Finds most relevant experiences from profile.

```python
# Input
profile: Dict[str, Any]
program_keywords: List[str]

# Output
[
    {"title": "Data Science Intern", "relevance_score": 0.9, ...},
    {"title": "Research Assistant", "relevance_score": 0.75, ...}
]
```

### **4. Requirement Checker** (`requirement_checker.py`)

Extracts program-specific requirements.

```python
# Input
program_info: Dict[str, Any]

# Output
{
    "must_mention": ["research experience", "programming skills"],
    "word_count": {"min": 500, "max": 800},
    "style_requirements": ["formal", "specific examples"]
}
```

---

## 🧠 Reflexion Memory System

The system uses **Reflexion** to learn from past generations:

### **Memory Features**

- **Successful Patterns**: Tracks strategies that improved scores
- **Dimension Patterns**: Per-dimension improvement strategies
- **Issue Resolutions**: Common problems and how they were fixed
- **Score-Range Matching**: Uses patterns from similar starting scores

### **Memory Storage**

```python
# Location
out/writing_agent_cache/reflexion_memory.json

# Structure
{
    "successful_patterns": [
        {
            "document_type": "personal_statement",
            "initial_score": 0.72,
            "final_score": 0.89,
            "improvement": 0.17,
            "strategies": ["Add specific numbers", "Reference program courses"],
            "dimension_improvements": {"personalization": 0.15, "program_alignment": 0.12},
            "score_range": "medium"
        }
    ],
    "dimension_patterns": {
        "personal_statement:personalization": [
            {"improvement": 0.15, "strategies": ["Add quantifiable metrics"]}
        ]
    }
}
```

### **Memory Integration**

```python
# Revise node uses memory suggestions
memory = get_memory()
patterns = memory.get_relevant_patterns("personal_statement", current_score=0.72)
suggestions = memory.suggest_strategies("personal_statement", ["Add specific examples"])

# Suggestions based on past successes
# e.g., "From past success: Add specific numbers to experience descriptions"
```

---

## 📊 WritingState Schema

The state flows through all nodes:

```python
class WritingState(TypedDict):
    # ===== Input =====
    profile: Dict[str, Any]
    program_info: Dict[str, Any]
    document_type: DocumentType
    corpus: Optional[Dict[str, str]]
    
    # ===== Configuration =====
    max_iterations: int
    quality_threshold: float
    llm_provider: str
    model_name: str
    temperature: float
    
    # ===== RAG Results =====
    retrieved_chunks: List[str]
    matched_experiences: List[Dict]
    program_keywords: List[str]
    
    # ===== Tool Results =====
    match_score: Optional[float]
    required_keywords: List[str]
    special_requirements: Dict[str, Any]
    tool_call_history: List[Dict]
    
    # ===== Generation =====
    plan: Optional[str]
    current_draft: Optional[str]
    draft_history: List[str]
    
    # ===== Reflection =====
    reflection_scores: List[ReflectionScore]
    overall_quality_score: float
    reflection_feedback: str
    improvement_suggestions: List[str]
    dimension_scores: Dict[str, float]
    weakest_dimensions: List[str]
    keyword_analysis: Dict[str, Any]
    
    # ===== Control Flow =====
    current_iteration: int
    iteration_logs: List[IterationLog]
    should_revise: bool
    is_complete: bool
```

---

## 🔄 Iteration Flow Details

### **Initial Generation**

```
Plan → RAG → React (generate) → Reflect
                                    │
                    Score: 0.72 (< 0.85 threshold)
                                    │
                                    ▼
                              Revise → Reflect
                                          │
                          Score: 0.89 (>= 0.85 threshold)
                                          │
                                          ▼
                                      Finalize
```

### **Adaptive Weighting**

The reflect node adjusts dimension weights based on:

1. **Current Score Level**: Different weights for low/medium/high scores
2. **Weakest Dimensions**: Focus on dimensions that need most improvement
3. **Stagnation Detection**: Stop if improvement < 0.02

```python
# Example adaptive weights for low score (< 0.70)
{
    "keyword_coverage": 0.25,    # Higher priority
    "personalization": 0.20,
    "coherence": 0.20,
    "program_alignment": 0.25,  # Higher priority
    "persuasiveness": 0.10
}
```

### **Stopping Conditions**

1. **Quality Met**: `overall_score >= quality_threshold`
2. **Max Iterations**: `current_iteration >= max_iterations`
3. **Stagnation**: Improvement < 0.02 for 2 consecutive iterations

---

## 🐛 Troubleshooting

### **Import Errors**

```bash
pip install langchain langgraph langchain-openai langchain-anthropic
```

### **API Key Errors**

```bash
# Verify .env file
cat .env

# Test API key
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"
```

### **Low Quality Scores**

1. Increase `max_iterations` (e.g., 5)
2. Lower `quality_threshold` (e.g., 0.75)
3. Provide more detailed profile (experiences, specific goals)
4. Use more powerful model (gpt-4 vs gpt-3.5-turbo)

### **Slow Generation**

- Normal: 10-20 seconds for high-quality output
- Use `gpt-3.5-turbo` for faster drafts
- Reduce `max_iterations`
- Reduce `retrieval_topk`

### **Memory Issues**

```python
# Clear memory cache
import os
os.remove("out/writing_agent_cache/reflexion_memory.json")
```

---

## 🔌 Extension Points

### **Adding New Tools**

Create a new tool in `tools/`:

```python
# tools/my_new_tool.py
from langchain.tools import tool

@tool
def my_new_tool(input_param: str) -> dict:
    """Tool description for the LLM"""
    # Implementation
    return {"result": "..."}

# Direct function version for node use
def my_new_tool_direct(input_param: str) -> dict:
    """Direct function without @tool decorator"""
    return {"result": "..."}
```

Register in `tools/__init__.py` and use in `react_node.py`.

### **Adding New Nodes**

Create a new node in `nodes/`:

```python
# nodes/my_new_node.py
from typing import Dict, Any
from ..state import WritingState

def my_new_node(state: WritingState) -> Dict[str, Any]:
    """Node implementation"""
    # Process state
    result = process_something(state)
    
    # Return updated state fields
    return {
        "my_new_field": result,
        "generation_metadata": {
            **state.get("generation_metadata", {}),
            "my_node_completed": True
        }
    }
```

Add to graph in `graph.py`:

```python
workflow.add_node("my_node", my_new_node)
workflow.add_edge("previous_node", "my_node")
```

### **Adding New Document Types**

1. Add to `state.py`:
```python
class DocumentType(str, Enum):
    PERSONAL_STATEMENT = "personal_statement"
    RESUME_BULLETS = "resume_bullets"
    RECOMMENDATION_LETTER = "recommendation_letter"
    COVER_LETTER = "cover_letter"  # New
```

2. Create prompts in `prompts/`:
```python
# prompts/cover_letter_prompts.py
def get_cover_letter_generation_prompt(...): ...
def get_cover_letter_revision_prompt(...): ...
```

3. Update `react_node.py` and `revise_node.py` to handle new type

---

## 📈 Performance Tips

1. **Use RAG Corpus**: Set `use_corpus=True` for better context
2. **Profile Detail**: More experiences = better personalization
3. **Specific Goals**: Detailed goals improve program alignment
4. **Iteration Count**: 2-3 iterations usually sufficient
5. **Model Selection**: GPT-4 for final, GPT-3.5 for drafts

---

## 📞 Support

- **API Docs**: `http://localhost:8000/docs`
- **Health Check**: `GET /health`
- **System Info**: `GET /systems/info`

---

**Version**: 2.0  
**Last Updated**: 2025-12-11  
**Status**: Production Ready ✅
