# Writing Agent v2.0 - Quick Start Guide

## 🎉 New System Features

The new Writing Agent has been fully implemented with significant improvements over the old system:

### ✨ Core Advantages

1. **Intelligent Generation, Not Template Filling**
   - Uses advanced LLMs like GPT-4/Claude for deep content understanding
   - No longer relies on fixed templates and simple if-else logic

2. **Multi-Dimensional Quality Assessment**
   - 5-dimension automatic scoring (keywords, personalization, coherence, program alignment, persuasiveness)
   - LLM self-reflection and improvement suggestions

3. **Iterative Optimization Mechanism**
   - Automatic multi-round improvements until quality standards are met
   - Learns from historical experience (Reflexion memory)

4. **Advanced AI Workflow**
   - RAG: Retrieve relevant program information
   - ReAct: Tool calling and reasoning
   - Reflection: Self-assessment
   - ReWOO: Plan-Tool-Solve

## 📦 Installation Steps

### 1. Install Dependencies

```bash
cd D:\DataWorkspace\DS301_Project\College_Application_Agent
pip install -r requirements.txt
```

Main new dependencies:
- `langchain>=0.1.0`
- `langgraph>=0.0.40`
- `langchain-openai>=0.0.5`
- `openai>=1.10.0`
- `faiss-cpu>=1.7.4`

### 2. Configure API Keys

Create a `.env` file (based on `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` and configure at least one LLM provider:

```env
# Use OpenAI (Recommended)
WRITING_AGENT_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4-turbo-preview

# Or use Qwen
WRITING_AGENT_LLM_PROVIDER=qwen
QWEN_API_KEY=your-qwen-key
QWEN_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-turbo
```

### 3. Start Service

```bash
python -m src.rag_service.api
```

Service will start at `http://localhost:8000`

## 🚀 Usage Examples

### API Call Examples

#### Generate Personal Statement

```python
import requests
import json

url = "http://localhost:8000/generate/writing-agent"

data = {
    "profile": {
        "name": "Alice Zhang",
        "major": "Data Science",
        "gpa": 3.78,
        "skills": ["Python", "Machine Learning", "Deep Learning", "SQL"],
        "experiences": [
            {
                "title": "Data Science Intern",
                "org": "Tech Company",
                "impact": "Built ML model improving prediction accuracy by 15%",
                "skills": ["Python", "TensorFlow", "Pandas"]
            },
            {
                "title": "Research Assistant",
                "org": "University Lab",
                "impact": "Analyzed large healthcare datasets and discovered key insights",
                "skills": ["R", "Statistical Analysis"]
            }
        ],
        "goals": "Apply machine learning to real-world business scenarios and become an expert in data science"
    },
    "program_text": "Columbia University Master's in Data Science program...(program description text)",
    "document_type": "personal_statement",
    "llm_provider": "openai",
    "max_iterations": 3,
    "quality_threshold": 0.85
}

response = requests.post(url, json=data)
result = response.json()

print("Generated Personal Statement:")
print(result["document"])
print("\nQuality Report:")
print(json.dumps(result["quality_report"], indent=2))
```

#### Generate Resume Bullets

```python
data = {
    "profile": { ... },  # Same as above
    "program_text": "...",
    "document_type": "resume_bullets",  # Change to resume_bullets
    "llm_provider": "openai",
    "max_iterations": 3
}

response = requests.post(url, json=data)
result = response.json()

print("Generated Resume Bullets:")
print(result["document"])
```

#### Generate Recommendation Letter

```python
data = {
    "profile": { ... },
    "program_text": "...",
    "document_type": "recommendation_letter",  # Change to recommendation_letter
    "llm_provider": "openai"
}

response = requests.post(url, json=data)
result = response.json()

print("Generated Recommendation Letter:")
print(result["document"])
```

## 📊 System Architecture

### File Structure

```
src/writing_agent/
├── __init__.py           # Module entry point
├── config.py             # Configuration management
├── state.py              # State definitions
├── graph.py              # LangGraph main graph
├── memory.py             # Reflexion memory
├── llm_utils.py          # LLM utility functions
├── nodes/                # Workflow nodes
│   ├── plan_node.py      # Planning node
│   ├── rag_node.py       # RAG retrieval node
│   ├── react_node.py     # Generation node
│   ├── reflect_node.py   # Reflection node
│   └── revise_node.py    # Revision node
├── tools/                # ReAct tools
│   ├── match_calculator.py    # Match degree calculation
│   ├── keyword_extractor.py   # Keyword extraction
│   ├── experience_finder.py   # Experience finder
│   └── requirement_checker.py # Requirement checker
└── prompts/              # Prompt templates
    ├── ps_prompts.py     # Personal Statement prompts
    ├── resume_prompts.py # Resume prompts
    ├── rl_prompts.py     # Recommendation letter prompts
    └── reflection_prompts.py # Reflection prompts
```

### Execution Flow

```
1. Plan Node
   ↓ Analyze task, formulate strategy
   
2. RAG Node
   ↓ Retrieve relevant information
   
3. ReAct Node (Generate)
   ↓ Call tools, generate initial draft
   
4. Reflect Node
   ↓ Multi-dimensional quality assessment
   
5. Check if standards are met
   ├─ Met → Finalize → End
   └─ Not Met → Revise Node → Back to Step 4
```

## 🎯 Configuration Parameters

### Main Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `llm_provider` | LLM Provider | `openai` | `openai` (Best quality) |
| `model_name` | Model name | `gpt-4-turbo-preview` | `gpt-4-turbo-preview` |
| `max_iterations` | Max iterations | `3` | `3-5` |
| `quality_threshold` | Quality threshold | `0.85` | `0.80-0.85` |
| `temperature` | Generation temperature | `0.7` | `0.6-0.8` |

### Quality Assessment Dimensions

1. **Keyword Coverage (20%)**: Coverage of key terms
2. **Personalization (25%)**: Degree of personalization
3. **Coherence (20%)**: Logical coherence
4. **Program Alignment (20%)**: Program match degree
5. **Persuasiveness (15%)**: Persuasive power

Total score ≥ 0.85 is considered passing

## 💡 Usage Tips

### 1. Provide Detailed Profile Information

The more detailed the profile information, the higher the generation quality:

```python
"experiences": [
    {
        "title": "Specific position",
        "org": "Organization name",
        "impact": "Specific achievements, preferably with numbers (e.g., improved by 15%)",
        "skills": ["Specific skills used"]
    }
]
```

### 2. Adjust Parameters by Importance

**Important Applications (e.g., PhD, Top Programs)**:
- `max_iterations`: 5
- `quality_threshold`: 0.90
- `model_name`: "gpt-4"

**General Applications**:
- `max_iterations`: 3
- `quality_threshold`: 0.85
- `model_name`: "gpt-4-turbo-preview"

**Quick Draft**:
- `max_iterations`: 2
- `quality_threshold`: 0.75
- `model_name`: "gpt-3.5-turbo"

### 3. Review Quality Report

Each generation returns a quality report:

```json
{
  "final_score": 0.89,
  "total_iterations": 2,
  "iteration_history": [...],
  "approved": true
}
```

If `final_score` is lower than expected, you can:
- Increase `max_iterations`
- Lower `quality_threshold`
- Enrich profile information
- Use a more powerful model

## 🔧 Troubleshooting

### Issue 1: ImportError: No module named 'langchain'

**Solution**:
```bash
pip install langchain langgraph langchain-openai
```

### Issue 2: API Key Error

**Solution**:
1. Check if `.env` file exists and is configured correctly
2. Verify that the API key is valid
3. Ensure environment variables are loaded correctly

### Issue 3: Low Generation Quality

**Solution**:
1. Increase iterations: `max_iterations=5`
2. Use a more powerful model: `model_name="gpt-4"`
3. Provide more detailed profile information
4. Check if program_text is detailed enough

### Issue 4: Slow Generation Speed

**Explanation**: This is normal
- Each iteration requires 2-3 LLM API calls
- GPT-4 response time is typically 2-5 seconds
- 3 iterations take approximately 10-20 seconds

To speed up:
- Reduce number of iterations
- Use a faster model (e.g., gpt-3.5-turbo)
- Lower quality threshold

## 📈 Comparison with Old System

| Feature | Old System | New System Writing Agent v2.0 |
|---------|------------|-------------------------------|
| Generation Method | Template filling | LLM deep generation |
| Quality Control | Keyword checking | 5-dimension LLM assessment |
| Improvement Mechanism | Simple replacement | Intelligent iterative optimization |
| Personalization | Low | High |
| Persuasiveness | Medium | High |
| Flexibility | Poor | Excellent |
| Speed | Fast (<1s) | Medium (10-20s) |
| Cost | Free | API fees |
| Quality | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🎓 Advanced Features

### 1. Use Your Own Corpus

```python
# Prepare your project documentation corpus
my_corpus = {
    "chunk_1": "Course description...",
    "chunk_2": "Program features...",
    # ...
}

# In the request, don't use program_text, instead pass through corpus
# (API endpoint needs to be modified to support corpus upload)
```

### 2. Customize Prompt Templates

Modify template files in `src/writing_agent/prompts/` to customize generation style.

### 3. Add New Tools

Create new tools in `src/writing_agent/tools/`:

```python
from langchain.tools import tool

@tool
def my_custom_tool(input: str) -> dict:
    """Tool description"""
    # Implementation logic
    return {"result": "..."}
```

## 📞 Support

If you have questions, please:
1. Check `src/writing_agent/README.md` for detailed documentation
2. Review log output
3. Create an issue on GitHub repo

## 🚀 Next Steps

1. Test basic functionality
2. Adjust configuration parameters
3. Compare effects with old system
4. Customize prompts as needed
5. Collect feedback for continuous improvement

Enjoy using the system!
