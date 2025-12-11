# 📊 Program Matching Service V2

## Overview

The **Program Matching Service** is an intelligent system that analyzes student profiles and matches them with the most suitable graduate programs. It supports **dual datasets** (V1 Legacy and V2 Enhanced) and uses a **multi-dimensional scoring algorithm** powered by AI, semantic similarity, and optional LLM-enhanced explanations.

---

## 🆕 V2 Highlights

- **6-Dimension Matching**: New **Curriculum** dimension for course-level analysis
- **Course-Level Analysis**: Match skills against specific program courses with descriptions
- **LLM-Powered Fit Reasons**: Personalized "Why This Program Fits You" explanations
- **Batch Processing**: Efficient batch LLM calls for multiple programs
- **Enhanced V2 Schema**: Rich structured data (courses, requirements, outcomes, background)

---

## 🎯 Key Features

### **Multi-Dimensional Scoring (V2)**

| Dimension | Weight | What It Evaluates |
|-----------|--------|-------------------|
| **Academic** | 25% | GPA match, major relevance, coursework alignment |
| **Skills** | 20% | Required skills coverage, course-extracted skills, skill breadth |
| **Experience** | 15% | Experience count, relevance, research vs professional orientation |
| **Goals** | 20% | Career goals alignment, mission match, career paths |
| **Requirements** | 10% | GPA requirement, tests, prerequisites compliance |
| **Curriculum** | 10% | Course-level skill matching, curriculum fit (V2 only) |

### **Match Quality Levels**

| Level | Score Range | Description |
|-------|-------------|-------------|
| 🟢 **Excellent** | 0.85+ | Strong match across all dimensions |
| 🟡 **Good** | 0.70-0.85 | Good overall fit with minor gaps |
| 🟠 **Moderate** | 0.55-0.70 | Reasonable fit, some areas to strengthen |
| 🟤 **Fair** | 0.40-0.55 | Partial match, significant preparation needed |
| 🔴 **Weak** | <0.40 | Limited alignment |

### **Intelligent Features**

- **BERT-based semantic similarity** via sentence-transformers
- **Configurable dimension weights** for personalized matching
- **LLM-enhanced fit reasons** (optional, uses GPT-4/Claude/Qwen)
- **Batch fit reason generation** for efficiency
- **Course description analysis** (V2)
- **Research vs Professional orientation matching**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Program Matching Service                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐                    ┌──────────────────────┐ │
│  │  V1 Matcher    │                    │    V2 Matcher        │ │
│  │  (Legacy)      │                    │    (Enhanced)        │ │
│  │  - 5 dims      │                    │    - 6 dims          │ │
│  │  - Basic data  │                    │    - Rich courses    │ │
│  └────────┬───────┘                    └──────────┬───────────┘ │
│           │                                       │              │
│           └──────────────┬────────────────────────┘              │
│                          ▼                                       │
│                 ┌────────────────────┐                          │
│                 │  DimensionScorer   │                          │
│                 │  (V1 / V2)         │                          │
│                 │  ├── Academic      │                          │
│                 │  ├── Skills        │                          │
│                 │  ├── Experience    │                          │
│                 │  ├── Goals         │                          │
│                 │  ├── Requirements  │                          │
│                 │  └── Curriculum*   │  *V2 only               │
│                 └─────────┬──────────┘                          │
│                           ▼                                      │
│                 ┌────────────────────┐                          │
│                 │  MatchExplainer    │                          │
│                 │  (V1 / V2)         │                          │
│                 │  ├── Rule-based    │                          │
│                 │  └── LLM-powered   │                          │
│                 └────────────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 Module Structure

```
src/matching_service/
├── __init__.py              # Module exports
├── README.md                # This documentation
│
├── # V1 (Legacy) Components
├── models.py                # V1 data models (Pydantic/dataclasses)
├── scorer.py                # V1 5-dimension scoring algorithms
├── matcher.py               # V1 core matching engine
├── explainer.py             # V1 LLM-enhanced explanations
│
├── # V2 (Enhanced) Components
├── models_v2.py             # V2 data models (6 dimensions, rich schema)
├── scorer_v2.py             # V2 scoring with curriculum analysis
├── matcher_v2.py            # V2 matching engine with batch processing
└── explainer_v2.py          # V2 LLM-powered personalized fit reasons
```

---

## 📖 API Endpoints

### **V1 Endpoints (Legacy Dataset)**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/match/programs` | POST | Match programs using V1 corpus |
| `/match/info` | GET | V1 matching service info |
| `/match/program/{id}/details` | GET | Get V1 program details |
| `/match/programs/list` | GET | List all V1 programs |
| `/match/compare` | POST | Compare specific V1 programs |

### **V2 Endpoints (Enhanced Dataset)**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2/match/programs` | POST | Match programs using V2 dataset |
| `/v2/match/info` | GET | V2 matching service info |
| `/v2/match/program/{id}/details` | GET | Get V2 program details with courses |
| `/v2/match/programs/list` | GET | List all V2 programs |

---

## 📖 Usage Examples

### **V2 Program Matching (Recommended)**

```python
import requests

url = "http://localhost:8000/v2/match/programs"

payload = {
    "profile": {
        "name": "Alice Zhang",
        "major": "Computer Science",
        "gpa": 3.8,
        "skills": ["Python", "Machine Learning", "SQL", "PyTorch", "TensorFlow"],
        "courses": ["Machine Learning", "Algorithms", "Statistics", "Deep Learning"],
        "experiences": [
            {
                "title": "Data Scientist Intern",
                "org": "TechCorp",
                "impact": "Built ML model improving accuracy by 15%",
                "skills": ["Python", "TensorFlow"]
            },
            {
                "title": "Research Assistant",
                "org": "University AI Lab",
                "impact": "Published paper on NLP with 50+ citations",
                "skills": ["Python", "NLP", "PyTorch"]
            }
        ],
        "goals": "Apply ML to solve real-world problems and lead AI research teams"
    },
    "top_k": 10,
    "min_score": 0.5,
    "include_curriculum_analysis": True,
    "include_course_recommendations": True
}

response = requests.post(url, json=payload, timeout=120)
result = response.json()

for match in result['matches'][:3]:
    print(f"\n{match['university']} - {match['program_name']}")
    print(f"  Overall Score: {match['overall_score']:.2f}")
    print(f"  Match Level: {match['match_level']}")
    
    # Dimension scores
    for dim, score_info in match['dimension_scores'].items():
        score = score_info['score'] if isinstance(score_info, dict) else score_info
        print(f"  {dim}: {score:.2f}")
    
    # Personalized fit reasons
    if match.get('fit_reasons'):
        print("  Why This Program Fits You:")
        for reason in match['fit_reasons'][:2]:
            print(f"    • {reason}")
    
    # Matched courses (V2)
    if match.get('matched_courses'):
        print(f"  Matched Courses: {', '.join(match['matched_courses'][:3])}")
```

### **Custom Dimension Weights**

```python
payload = {
    "profile": { ... },
    "custom_weights": {
        "academic": 0.30,      # Emphasize academic fit
        "skills": 0.25,
        "experience": 0.10,    # Less weight on experience
        "goals": 0.20,
        "requirements": 0.05,
        "curriculum": 0.10     # V2 only
    }
}
```

### **Response Structure (V2)**

```json
{
    "success": true,
    "message": "Found 10 matching programs",
    "matches": [
        {
            "program_id": "seas.harvard.edu/12345",
            "program_name": "MS in Data Science",
            "university": "Harvard University",
            "department": "School of Engineering and Applied Sciences",
            "overall_score": 0.87,
            "match_level": "excellent",
            "dimension_scores": {
                "academic": {"score": 0.92, "weight": 0.25, "matched_items": ["GPA: 3.80", "Major: Computer Science"]},
                "skills": {"score": 0.85, "weight": 0.20, "matched_items": ["Skill: Python", "Skill: Machine Learning"]},
                "experience": {"score": 0.88, "weight": 0.15, "matched_items": ["Research experience"]},
                "goals": {"score": 0.83, "weight": 0.20, "matched_items": ["Career goals alignment"]},
                "requirements": {"score": 0.90, "weight": 0.10, "matched_items": ["Meets GPA requirement"]},
                "curriculum": {"score": 0.78, "weight": 0.10, "matched_items": ["Course: Advanced Machine Learning"]}
            },
            "strengths": [
                "GPA 3.80 significantly exceeds typical requirement",
                "Possess 8/10 required skills",
                "Research experience aligns with program's research focus"
            ],
            "gaps": [
                "Consider strengthening statistics background"
            ],
            "fit_reasons": [
                "🎓 Your Computer Science background provides strong preparation for this program's curriculum",
                "💡 Your skills in Python, Machine Learning, PyTorch directly apply to the program's focus",
                "🚀 The program's career outcomes align with your goal to lead AI research teams"
            ],
            "matched_courses": ["Advanced Machine Learning", "Deep Learning", "Statistical Inference"],
            "relevant_courses": ["AI Ethics", "Data Visualization"],
            "recommendations": [
                "Profile is strong - focus on crafting compelling application materials",
                "Research the curriculum and express interest in specific courses"
            ],
            "metadata": {
                "department": "School of Engineering and Applied Sciences",
                "source_url": "https://seas.harvard.edu/...",
                "has_courses": true,
                "has_career_info": true
            }
        }
    ],
    "total_programs_evaluated": 85,
    "overall_insights": {
        "strong_areas": ["academic", "skills"],
        "improvement_areas": ["curriculum"],
        "recommendation": "Your profile shows strong alignment with Data Science programs"
    },
    "dataset_version": "v2",
    "processing_time_seconds": 3.5
}
```

---

## 🔧 Configuration

### **Environment Variables**

| Variable | Description | Default |
|----------|-------------|---------|
| `V2_LLM_PROVIDER` | LLM provider for V2 explainer | `openai` |
| `OPENAI_API_KEY` | OpenAI API key (for LLM explanations) | Required |
| `ANTHROPIC_API_KEY` | Anthropic API key (alternative) | Optional |

### **Matcher Initialization**

```python
from src.matching_service import ProgramMatcherV2

# Initialize V2 matcher with LLM explainer
matcher = ProgramMatcherV2(
    corpus_dir="data_preparation/dataset/graduate_programs",
    use_llm_explainer=True,
    llm_provider="openai"
)

print(f"Loaded {len(matcher.programs)} programs")
```

---

## 📊 V2 Data Models

### **ProgramDataV2**

```python
@dataclass
class ProgramDataV2:
    # Core identifiers
    program_id: str
    source_url: str
    
    # Basic info
    program_name: str
    school: str
    department: str
    duration: str
    
    # Structured data
    courses: List[CourseInfo]                          # name + description
    application_requirements: ApplicationRequirementsV2
    program_background: ProgramBackgroundV2
    training_outcomes: TrainingOutcomesV2
    
    # Text content
    raw_text: str
    chunks: List[ChunkInfo]
```

### **DimensionScoreV2**

```python
@dataclass
class DimensionScoreV2:
    dimension: str      # "academic", "skills", etc.
    score: float        # 0.0 - 1.0
    weight: float       # Dimension weight
    details: Dict       # Detailed scoring breakdown
    contributing_factors: List[str]  # Human-readable factors
    matched_items: List[str]         # What matched
    missing_items: List[str]         # What's missing
```

### **MatchDimensionV2 Enum**

```python
class MatchDimensionV2(str, Enum):
    ACADEMIC = "academic"
    SKILLS = "skills"
    EXPERIENCE = "experience"
    GOALS = "goals"
    REQUIREMENTS = "requirements"
    CURRICULUM = "curriculum"  # V2 only
```

---

## 🧮 Scoring Algorithms (V2)

### **Academic Scoring**

```
Score = GPA Match (35%) + Major Relevance (35%) + Coursework Alignment (30%)

- GPA Match: Compares student GPA to program's typical requirement
- Major Relevance: Semantic similarity between student major and program field
- Coursework: Semantic matching of completed courses to program courses
```

### **Skills Scoring**

```
Score = Required Coverage (50%) + Course Skills (30%) + Breadth Bonus (20%)

- Required Coverage: % of required skills the student possesses
- Course Skills: Skills extracted from course descriptions
- Breadth Bonus: Additional relevant skills beyond requirements
```

### **Curriculum Scoring (V2 Only)**

```
Score = Course Name Match (40%) + Description Match (40%) + Coverage (20%)

- Analyzes each program course against student skills
- Uses course descriptions for deeper semantic matching
- Evaluates curriculum breadth coverage
```

### **Experience Scoring**

```
Score = Quantity (25%) + Relevance (45%) + Orientation Match (30%)

- Quantity: Number of experiences (3+ = full score)
- Relevance: Semantic similarity of experience descriptions to program
- Orientation: Research vs Professional experience vs program focus
```

### **Goals Scoring**

```
Score = Goals Alignment (50%) + Mission Match (30%) + Career Paths (20%)

- Goals Alignment: Semantic match of student goals to program outcomes
- Mission Match: Alignment with program mission/values
- Career Paths: Match with program's career path offerings
```

---

## 🤖 LLM-Powered Fit Reasons (V2)

The V2 explainer generates personalized fit reasons using LLM:

### **How It Works**

1. **Batch Processing**: Groups programs (batch_size=3) for efficiency
2. **Context Building**: Provides profile, program data, and dimension scores to LLM
3. **Personalized Generation**: LLM creates natural, specific fit reasons
4. **Fallback**: Rule-based generation if LLM fails

### **Example Fit Reasons**

```python
# LLM-generated (personalized, specific)
[
    "Your Machine Learning coursework at University of Michigan directly prepares you for the program's advanced ML curriculum, including courses like CS229 and CS231n",
    "Your research experience in NLP aligns perfectly with Professor Smith's lab, which focuses on transformer architectures",
    "The program's emphasis on industry partnerships matches your goal of applying ML to real-world problems"
]

# Rule-based fallback (template-based)
[
    "🎓 Your Computer Science background provides strong preparation for this program's curriculum",
    "💡 Your skills in Python, Machine Learning directly apply to the program's focus",
    "🚀 The program's career outcomes align with your career goals"
]
```

---

## 🔄 V1 vs V2 Comparison

| Feature | V1 (Legacy) | V2 (Enhanced) |
|---------|-------------|---------------|
| **Dimensions** | 5 | 6 (+ Curriculum) |
| **Corpus Location** | `data/corpus/` | `data_preparation/dataset/` |
| **Course Data** | Names only | Names + Descriptions |
| **Program Structure** | Flat sections | Rich nested fields |
| **Fit Reasons** | Rule-based | LLM-powered |
| **Application Requirements** | Basic | Structured (GRE, TOEFL, etc.) |
| **Training Outcomes** | Limited | Career paths, research focus |
| **Speed** | ~1-2s | ~3-5s (with LLM) |

---

## 🚀 Performance

| Metric | V1 | V2 |
|--------|----|----|
| **Response time (no LLM)** | 1-2s | 2-3s |
| **Response time (with LLM)** | 2-4s | 3-5s |
| **Programs supported** | 80+ | 80+ |
| **Batch LLM calls** | No | Yes (3 per batch) |
| **Corpus loading** | ~1s | ~2s |

---

## 🐛 Troubleshooting

### **V2 Matcher Not Loading**

```bash
# Check if dataset exists
ls data_preparation/dataset/graduate_programs/

# Verify in API logs
# Look for: "✅ V2 Program Matcher initialized with X programs"
```

### **LLM Explainer Errors**

```python
# Check environment variable
import os
print(os.getenv("V2_LLM_PROVIDER"))  # Should be "openai", "anthropic", or "qwen"
print(os.getenv("OPENAI_API_KEY"))    # Should be set

# Disable LLM explainer (use rule-based)
matcher = ProgramMatcherV2(use_llm_explainer=False)
```

### **Timeout Issues**

```python
# Increase request timeout for V2 with LLM
response = requests.post(url, json=payload, timeout=120)

# Or disable LLM in request
payload["use_llm_explanation"] = False
```

---

## 🔮 Future Enhancements

- [ ] Vector search for semantic course matching
- [ ] Program ranking integration (QS, US News)
- [ ] Application success rate tracking
- [ ] Multi-student batch matching
- [ ] Interactive weight tuning UI
- [ ] More granular curriculum analysis

---

## 📞 Support

- **API Docs**: `http://localhost:8000/docs`
- **Health Check**: `GET /health` 
- **Service Info**: `GET /v2/match/info`

---

**Version**: 2.0  
**Last Updated**: 2025-12-11  
**Status**: Production Ready ✅
