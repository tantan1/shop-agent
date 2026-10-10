"""工具选择标注 Web 薄层后端。

独立 FastAPI 进程:以 Langfuse(trace 原料)为输入,SQLite(标注真值)为输出,
不反向写 Langfuse score(Options A)。详见 docs/architecture/tool-select-annotation-web-layer-design.md。
"""
