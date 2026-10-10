# tool-select-annotator（工具选择标注 Web 薄层）

独立 FastAPI + Streamlit 服务,实现 `docs/architecture/tool-select-annotation-web-layer-design.md`。

## 职责
- **输入**:只读拉取 Langfuse `tool_select_review` trace。
- **真值**:本地 SQLite(`ANNOTATOR_SQLITE_DB`)保存 `gold_tool`、去重、自动银标闸门、用户反馈信号。
- **不回写 Langfuse score**(设计 Option A)。

## 目录
- `backend/` —— FastAPI 后端(index_store / langfuse_client / llm_prelabel / main)
- `frontend/app.py` —— Streamlit 标注台

## 运行
```bash
cd apps/tool-select-annotator
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # 填 Langfuse key + ANNOTATOR_SQLITE_DB
# 后端
uvicorn backend.main:app --host 0.0.0.0 --port 8137
# 前端(另开终端)
streamlit run frontend/app.py
```
打开前端 → 点「增量同步」拉新 trace → 在人工队列里逐条标注 / 一键确认,或看「统计」页观察自动银标占比。

## 复检规模化(§5.3)
同步阶段 `gate_auto_label` 对 `margin >= AUTO_PASS_MARGIN` 的样本自动银标(`auto_passed=1`,不进人工队列),
并抽 `AUTO_SPOTCHECK_RATE` 进抽样复核。关注统计页:自动银标占比高=减负有效;spotcheck 一致率低则收紧 `AUTO_PASS_MARGIN`。

## shop-agent 侧联动(可选,启用 SQLite 真值主路径)
在 shop-agent 配置 `MLOPS_TOOL_SELECT_GOLD_DB=<指向同一个 annotations_idx.db>` 后:
- `record_correction`(`/agent/correction`)会**双写**到该 SQLite(迁移期保留 Langfuse score 以兼容旧路径)。
- `export_and_train` 训练主路径改读该 SQLite 的 `gold_tool`(不再依赖 Langfuse score),见 `langfuse_mlops.py`。
- 前置:`capture_tool_select` 已多记 `top_tools`(top-k 分数)供边际闸门使用。
