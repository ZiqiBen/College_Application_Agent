# Writing Agent v2.0 - 快速开始指南

## 🎉 新系统特性

全新的Writing Agent已经完成实现，相比旧系统有巨大改进：

### ✨ 核心优势

1. **智能生成，非模板填充**
   - 使用GPT-4/Claude等先进LLM深度理解内容
   - 不再依赖固定模板和简单if-else逻辑

2. **多维度质量评估**
   - 5个维度自动评分（关键词、个性化、连贯性、匹配度、说服力）
   - LLM自我反思和改进建议

3. **迭代优化机制**
   - 自动多轮改进直到达到质量标准
   - 学习历史经验（Reflexion记忆）

4. **先进AI工作流**
   - RAG: 检索相关程序信息
   - ReAct: 工具调用和推理
   - Reflection: 自我评估
   - ReWOO: 规划-工具-解决

## 📦 安装步骤

### 1. 安装依赖

```bash
cd D:\DataWorkspace\DS301_Project\College_Application_Agent
pip install -r requirements.txt
```

主要新增依赖：
- `langchain>=0.1.0`
- `langgraph>=0.0.40`
- `langchain-openai>=0.0.5`
- `openai>=1.10.0`
- `faiss-cpu>=1.7.4`

### 2. 配置API密钥

创建`.env`文件（基于`.env.example`）：

```bash
cp .env.example .env
```

编辑`.env`，至少配置一个LLM provider：

```env
# 使用OpenAI (推荐)
WRITING_AGENT_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4-turbo-preview

# 或使用Qwen
WRITING_AGENT_LLM_PROVIDER=qwen
QWEN_API_KEY=your-qwen-key
QWEN_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-turbo
```

### 3. 启动服务

```bash
python -m src.rag_service.api
```

服务将在 `http://localhost:8000` 启动

## 🚀 使用示例

### API调用示例

#### 生成Personal Statement

```python
import requests
import json

url = "http://localhost:8000/generate/writing-agent"

data = {
    "profile": {
        "name": "张三",
        "major": "数据科学",
        "gpa": 3.78,
        "skills": ["Python", "机器学习", "深度学习", "SQL"],
        "experiences": [
            {
                "title": "数据分析实习生",
                "org": "某科技公司",
                "impact": "构建机器学习模型，将预测准确率提升15%",
                "skills": ["Python", "TensorFlow", "Pandas"]
            },
            {
                "title": "研究助理",
                "org": "大学实验室",
                "impact": "分析大型医疗数据集，发现关键洞察",
                "skills": ["R", "统计分析"]
            }
        ],
        "goals": "希望将机器学习应用于实际业务场景，成为数据科学领域的专家"
    },
    "program_text": "哥伦比亚大学数据科学硕士项目...(项目介绍文本)",
    "document_type": "personal_statement",
    "llm_provider": "openai",
    "max_iterations": 3,
    "quality_threshold": 0.85
}

response = requests.post(url, json=data)
result = response.json()

print("生成的Personal Statement:")
print(result["document"])
print("\n质量报告:")
print(json.dumps(result["quality_report"], indent=2, ensure_ascii=False))
```

#### 生成Resume Bullets

```python
data = {
    "profile": { ... },  # 同上
    "program_text": "...",
    "document_type": "resume_bullets",  # 改为resume_bullets
    "llm_provider": "openai",
    "max_iterations": 3
}

response = requests.post(url, json=data)
result = response.json()

print("生成的Resume Bullets:")
print(result["document"])
```

#### 生成Recommendation Letter

```python
data = {
    "profile": { ... },
    "program_text": "...",
    "document_type": "recommendation_letter",  # 改为recommendation_letter
    "llm_provider": "openai"
}

response = requests.post(url, json=data)
result = response.json()

print("生成的Recommendation Letter:")
print(result["document"])
```

## 📊 系统架构

### 文件结构

```
src/writing_agent/
├── __init__.py           # 模块入口
├── config.py             # 配置管理
├── state.py              # 状态定义
├── graph.py              # LangGraph主图
├── memory.py             # Reflexion记忆
├── llm_utils.py          # LLM工具函数
├── nodes/                # 工作流节点
│   ├── plan_node.py      # 规划节点
│   ├── rag_node.py       # RAG检索节点
│   ├── react_node.py     # 生成节点
│   ├── reflect_node.py   # 反思节点
│   └── revise_node.py    # 修订节点
├── tools/                # ReAct工具
│   ├── match_calculator.py    # 匹配度计算
│   ├── keyword_extractor.py   # 关键词提取
│   ├── experience_finder.py   # 经历查找
│   └── requirement_checker.py # 要求检查
└── prompts/              # Prompt模板
    ├── ps_prompts.py     # PS提示词
    ├── resume_prompts.py # 简历提示词
    ├── rl_prompts.py     # 推荐信提示词
    └── reflection_prompts.py # 反思提示词
```

### 执行流程

```
1. Plan Node
   ↓ 分析任务，制定策略
   
2. RAG Node
   ↓ 检索相关信息
   
3. ReAct Node (Generate)
   ↓ 调用工具，生成初稿
   
4. Reflect Node
   ↓ 多维度评估质量
   
5. 判断是否达标
   ├─ 达标 → Finalize → 结束
   └─ 未达标 → Revise Node → 回到步骤4
```

## 🎯 配置参数说明

### 主要参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `llm_provider` | LLM提供商 | `openai` | `openai` (质量最好) |
| `model_name` | 模型名称 | `gpt-4-turbo-preview` | `gpt-4-turbo-preview` |
| `max_iterations` | 最大迭代次数 | `3` | `3-5` |
| `quality_threshold` | 质量阈值 | `0.85` | `0.80-0.85` |
| `temperature` | 生成温度 | `0.7` | `0.6-0.8` |

### 质量评估维度

1. **Keyword Coverage (20%)**: 关键词覆盖度
2. **Personalization (25%)**: 个性化程度
3. **Coherence (20%)**: 逻辑连贯性
4. **Program Alignment (20%)**: 项目匹配度
5. **Persuasiveness (15%)**: 说服力

总分 ≥ 0.85 视为通过

## 💡 使用建议

### 1. Profile信息要详细

提供越详细的profile信息，生成质量越高：

```python
"experiences": [
    {
        "title": "具体职位",
        "org": "组织名称",
        "impact": "具体成就，最好有数字（如提升15%）",
        "skills": ["使用的具体技能"]
    }
]
```

### 2. 根据重要性调整参数

**重要申请（如PhD、顶尖项目）**：
- `max_iterations`: 5
- `quality_threshold`: 0.90
- `model_name`: "gpt-4"

**一般申请**：
- `max_iterations`: 3
- `quality_threshold`: 0.85
- `model_name`: "gpt-4-turbo-preview"

**快速草稿**：
- `max_iterations`: 2
- `quality_threshold`: 0.75
- `model_name`: "gpt-3.5-turbo"

### 3. 查看质量报告

每次生成都会返回质量报告：

```json
{
  "final_score": 0.89,
  "total_iterations": 2,
  "iteration_history": [...],
  "approved": true
}
```

如果`final_score`低于期望，可以：
- 增加`max_iterations`
- 降低`quality_threshold`
- 丰富profile信息
- 使用更强大的模型

## 🔧 故障排除

### 问题1: ImportError: No module named 'langchain'

**解决**：
```bash
pip install langchain langgraph langchain-openai
```

### 问题2: API密钥错误

**解决**：
1. 检查`.env`文件是否存在且正确配置
2. 验证API key是否有效
3. 确保环境变量被正确加载

### 问题3: 生成质量不高

**解决**：
1. 增加迭代次数：`max_iterations=5`
2. 使用更强模型：`model_name="gpt-4"`
3. 提供更详细的profile信息
4. 检查program_text是否足够详细

### 问题4: 生成速度慢

**说明**: 这是正常现象
- 每次迭代需要调用2-3次LLM API
- GPT-4响应时间通常2-5秒
- 3次迭代约需要10-20秒

如需加速：
- 减少迭代次数
- 使用更快的模型（如gpt-3.5-turbo）
- 降低质量阈值

## 📈 与旧系统对比

| 特性 | 旧系统 | 新系统 Writing Agent v2.0 |
|------|--------|---------------------------|
| 生成方式 | 模板填充 | LLM深度生成 |
| 质量控制 | 关键词检查 | 5维度LLM评估 |
| 改进机制 | 简单替换 | 智能迭代优化 |
| 个性化 | 低 | 高 |
| 说服力 | 中 | 高 |
| 灵活性 | 差 | 优秀 |
| 速度 | 快（<1秒） | 中（10-20秒） |
| 成本 | 免费 | API费用 |
| 质量 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🎓 高级功能

### 1. 使用自己的Corpus

```python
# 准备你的项目文档corpus
my_corpus = {
    "chunk_1": "项目课程描述...",
    "chunk_2": "项目特色介绍...",
    # ...
}

# 在请求中不使用program_text，而是通过corpus传入
# (需要修改API endpoint以支持corpus上传)
```

### 2. 自定义Prompt模板

修改 `src/writing_agent/prompts/` 中的模板文件来自定义生成风格。

### 3. 添加新工具

在 `src/writing_agent/tools/` 中创建新工具：

```python
from langchain.tools import tool

@tool
def my_custom_tool(input: str) -> dict:
    """工具描述"""
    # 实现逻辑
    return {"result": "..."}
```

## 📞 支持

如有问题，请：
1. 查看 `src/writing_agent/README.md` 详细文档
2. 检查日志输出
3. 在GitHub repo创建issue

## 🚀 下一步

1. 测试基本功能
2. 调整配置参数
3. 与旧系统对比效果
4. 根据需要自定义prompt
5. 收集反馈持续改进

祝使用愉快！
