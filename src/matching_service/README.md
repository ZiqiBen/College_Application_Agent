# 📊 Program Matching Service

## Overview

The **Program Matching Service** analyzes student profiles and matches them with the most suitable graduate programs using a multi-dimensional scoring algorithm powered by AI and semantic similarity.

---

## 🎯 Key Features

### **Multi-Dimensional Scoring**
Evaluates student-program fit across 5 dimensions:

| Dimension | Weight | Factors |
|-----------|--------|---------|
| **Academic** | 30% | GPA match, major relevance, coursework background |
| **Skills** | 25% | Required skills coverage, skill breadth |
| **Experience** | 20% | Experience quantity, relevance, impact |
| **Goals** | 15% | Career goal alignment, clarity |
| **Requirements** | 10% | GPA, language tests, prerequisites compliance |

### **Intelligent Matching**
- **BERT-based semantic similarity** for major/experience/goals alignment
- **Configurable weights** for different matching priorities
- **LLM-enhanced explanations** (optional) for personalized insights
- **Automatic filtering** by tuition, location, or other criteria

### **Comprehensive Results**
- **Ranked programs** with overall match scores (0-1)
- **Strengths & gaps** analysis for each program
- **Actionable recommendations** for profile improvement
- **Match quality levels**: Excellent (0.8+), Good (0.6-0.8), Moderate (0.4-0.6), Weak (<0.4)

---

## 🏗️ Architecture

\`\`\`
┌─────────────────────────────────────────┐
│      Program Matching Service           │
├─────────────────────────────────────────┤
│                                          │
│  ┌────────────┐      ┌──────────────┐  │
│  │  Corpus    │─────▶│   Matcher    │  │
│  │  Loader    │      │   Engine     │  │
│  └────────────┘      └──────────────┘  │
│                             │            │
│                             ▼            │
│                   ┌──────────────────┐  │
│                   │ Dimension Scorer │  │
│                   │  (5 dimensions)  │  │
│                   └──────────────────┘  │
│                             │            │
│                             ▼            │
│                   ┌──────────────────┐  │
│                   │  LLM Explainer   │  │
│                   │   (Optional)     │  │
│                   └──────────────────┘  │
│                                          │
└─────────────────────────────────────────┘
\`\`\`

---

## 📖 Usage

### **API Endpoints**

#### **1. Match Programs**
\`\`\`bash
POST /match/programs
\`\`\`

**Request:**
\`\`\`json
{
  "profile": {
    "name": "Alice Zhang",
    "major": "Computer Science",
    "gpa": 3.8,
    "skills": ["Python", "Machine Learning", "SQL"],
    "courses": ["Algorithms", "Statistics"],
    "experiences": [
      {
        "title": "Data Scientist Intern",
        "org": "TechCorp",
        "impact": "Built ML model improving accuracy by 15%",
        "skills": ["Python", "TensorFlow"]
      }
    ],
    "goals": "Apply ML to solve real-world problems in AI research"
  },
  "top_k": 5,
  "min_score": 0.6,
  "use_llm_explanation": true
}
\`\`\`

**Response:**
\`\`\`json
{
  "success": true,
  "message": "Found 5 matching programs",
  "matches": [
    {
      "program_id": "stanford-mscs",
      "program_name": "MS in Computer Science",
      "university": "Stanford University",
      "overall_score": 0.87,
      "match_level": "excellent",
      "dimension_scores": {
        "academic": {"score": 0.92, "weight": 0.30},
        "skills": {"score": 0.85, "weight": 0.25},
        "experience": {"score": 0.88, "weight": 0.20},
        "goals": {"score": 0.83, "weight": 0.15},
        "requirements": {"score": 0.90, "weight": 0.10}
      },
      "strengths": [
        "Strong academic match (92%)",
        "Excellent GPA (3.80)",
        "Substantial professional experience"
      ],
      "gaps": [],
      "recommendations": [
        "Profile is strong - focus on crafting compelling application materials"
      ],
      "explanation": "Outstanding fit! Your background strongly aligns with..."
    }
  ],
  "total_programs_evaluated": 25,
  "processing_time_seconds": 2.3
}
\`\`\`

---

#### **2. Get Service Info**
\`\`\`bash
GET /match/info
\`\`\`

Returns available programs count, dimensions, weights, and example usage.

---

#### **3. Compare Programs**
\`\`\`bash
POST /match/compare
\`\`\`

Compare specific programs side-by-side for a given student profile.

**Request:**
\`\`\`json
{
  "profile": { ... },
  "program_ids": ["stanford-mscs", "mit-eecs", "cmu-ml"]
}
\`\`\`

---

## 🔧 Configuration

### **Custom Weights**
Adjust dimension importance based on priorities:

\`\`\`json
{
  "profile": { ... },
  "custom_weights": {
    "academic": 0.40,
    "skills": 0.30,
    "experience": 0.15,
    "goals": 0.10,
    "requirements": 0.05
  }
}
\`\`\`

### **Filters**
Apply constraints to narrow results:

\`\`\`json
{
  "profile": { ... },
  "filters": {
    "max_tuition": 50000,
    "location": "California"
  }
}
\`\`\`

---

## 📂 File Structure

\`\`\`
src/matching_service/
├── __init__.py          # Module exports
├── models.py            # Data models (Pydantic/dataclasses)
├── scorer.py            # Multi-dimensional scoring algorithms
├── matcher.py           # Core matching engine
└── explainer.py         # LLM-enhanced explanations
\`\`\`

---

## 🧪 Example Usage (Python)

\`\`\`python
import requests

url = "http://localhost:8000/match/programs"

profile = {
    "name": "John Doe",
    "major": "Data Science",
    "gpa": 3.7,
    "skills": ["Python", "R", "Machine Learning"],
    "experiences": [
        {
            "title": "Research Assistant",
            "org": "University Lab",
            "impact": "Published paper on NLP with 50+ citations",
            "skills": ["Python", "NLP"]
        }
    ],
    "goals": "Pursue PhD in AI and contribute to cutting-edge research"
}

response = requests.post(url, json={
    "profile": profile,
    "top_k": 3,
    "min_score": 0.7,
    "use_llm_explanation": True
})

result = response.json()

for match in result['matches']:
    print(f"\\n{match['program_name']} - {match['university']}")
    print(f"Match Score: {match['overall_score']:.1%}")
    print(f"Level: {match['match_level']}")
    print(f"Strengths: {', '.join(match['strengths'][:2])}")
    print(f"Explanation: {match['explanation'][:100]}...")
\`\`\`

---

## 🎓 How It Works

1. **Load Programs**: Reads program data from \`data/corpus/\` (JSON files)
2. **Parse Profile**: Validates and structures student information
3. **Score Dimensions**: Calculates 5 dimension scores using:
   - **Rule-based matching** (GPA, courses, skills)
   - **BERT embeddings** (semantic similarity for major, experience, goals)
4. **Compute Overall Score**: Weighted average of dimension scores
5. **Rank & Filter**: Sort by score, apply filters, return top K
6. **Generate Explanations**: 
   - **Rule-based**: Fast, deterministic explanations
   - **LLM-enhanced**: Personalized, context-aware insights (if enabled)

---

## 📊 Quality Metrics

- **Score Range**: 0.0 (no match) to 1.0 (perfect match)
- **Threshold Recommendations**:
  - **0.8+**: Excellent fit - highly recommended
  - **0.6-0.8**: Good fit - strong candidate
  - **0.4-0.6**: Moderate fit - consider improvements
  - **<0.4**: Weak fit - significant preparation needed

---

## 🚀 Performance

- **Average response time**: 1-3 seconds (depends on corpus size)
- **Corpus loading**: One-time on server startup
- **LLM enhancement**: +1-2 seconds per program (if enabled)
- **Scalability**: Handles 100+ programs efficiently

---

## 🔮 Future Enhancements

- [ ] Add program ranking integration (QS, US News)
- [ ] Support batch matching for multiple students
- [ ] Historical matching data & success rate tracking
- [ ] Machine learning model fine-tuning on application outcomes
- [ ] Interactive visualization dashboard

---

## 📞 Support

For issues or questions:
- Check API docs: \`http://localhost:8000/docs\`
- Review logs for debugging
- Ensure corpus is properly loaded in \`data/corpus/\`

---

**Version**: 1.0.0  
**Last Updated**: 2024-12  
**Status**: Production Ready ✅
