```markdown
# 高校官网语料库建设 — 精简高效实施方案（可运行 PoC）

说明
- 这是一个最小可运行的 PoC 管线，按我们讨论的精简流程实现：
  1. Discovery（可选，手工 seeds.txt / 可接入 ReAct agent）
  2. Snapshot（下载并保存原始快照）
  3. Normalize + Prefilter（HTML/PDF 提取与轻量过滤）
  4. Chunk（文本切片）
  5. Aggregated Extraction（调用 Qwen 抽取结构化 JSON）
  6. Validate & Store（规则校验并存 JSON，保存溯源与 LLM metadata）
- 代码轻量、便于扩展。Qwen 适配器为通用 HTTP 风格：请根据你使用的 Qwen 服务调整 `QWEN_API_URL`/`QWEN_API_KEY` 或替换适配器函数。

快速开始
1. 环境
   - Python 3.9+
   - 安装依赖：
     pip install -r requirements.txt

2. 配置（环境变量）
   - QWEN_API_URL e.g. https://api.qwen.example/v1/chat/completions
   - QWEN_API_KEY

3. 准备 seeds.txt（每行一个 URL），示例见 seeds_example.txt。

4. 运行 PoC（会把输出写入 data/corpus/ 与 out/ 目录）：
   python pipeline.py seeds.txt

主要文件
- pipeline.py          : 主执行脚本（crawl -> extract -> call Qwen -> validate -> save）
- qwen_adapter.py      : Qwen HTTP 调用适配器（替换为真实 SDK/endpoint）
- utils.py             : 抓取、快照、HTML/PDF 提取、chunk、embedding helper
- schema.py            : Pydantic schema（CorpusDoc / ExtractedFields / Chunk / Snippet / LLMCall）
- requirements.txt     : 依赖列表
- seeds_example.txt    : seeds 示例

注意
- 生产使用前请完善 robots.txt 检查、并发/限速、代理与反爬策略；Playwright 仅在需渲染时启用。
- 本代码为 PoC，Qwen 返回解析与稳定性请在实际平台上根据 API 文档微调解析路径与调用参数。
```