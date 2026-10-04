"""
合并 AI 编码系列所有文章为一个文件。
用法: python scripts/build_full_series.py
输出: docs/ai-coding-series/ai-coding-series-full.md
"""

import re
from pathlib import Path

# 本脚本位于 apps/shop-agent/scripts/train/sft/，仓库根需上溯 5 层
REPO_ROOT = Path(__file__).resolve().parents[5]
SERIES_DIR = REPO_ROOT / "docs" / "ai-coding-series"
OUTPUT_FILE = SERIES_DIR / "ai-coding-series-full.md"

# SDLC 自然流程顺序（与 structure.md 保持一致）
# (stage_label, filename, short_title)
ARTICLES = [
    ("入门", "01-三层漏斗总览.md", "AI 编码的三层漏斗"),
    ("入门", "02-AI编码的边界.md", "AI 编码的边界"),
    ("准备", "03-Prompt工程实战.md", "Prompt 工程实战"),
    ("准备", "04-规则即代码.md", "规则即代码：用 Rules 自动遵守团队规范"),
    ("准备", "05-精准输入.md", "精准输入：@ 引用与搜索后生成"),
    ("准备", "06-模式选择矩阵.md", "Ask/Craft/Plan 模式选择矩阵"),
    ("准备", "07-SDD规约驱动开发.md", "SDD 规约驱动开发"),
    ("设计", "08-系统架构设计.md", "AI 辅助系统架构设计"),
    ("设计", "09-API与数据库设计.md", "AI 辅助的 API 与数据库设计"),
    ("编码", "10-Agent协作流水线.md", "多 Agent 协作流水线"),
    ("编码", "11-AI辅助调试.md", "AI 辅助调试"),
    ("编码", "21-代码理解取舍：为什么我们没上 Code RAG.md",
     "代码理解取舍：为什么我们没上 Code RAG"),
    ("质量", "12-代码质量与静态分析.md", "AI 驱动的代码质量与静态分析"),
    ("质量", "13-分层测试.md", "分层测试的 AI 辅助"),
    ("质量", "14-性能优化.md", "AI 辅助性能优化"),
    ("质量", "15-AI辅助代码审查.md", "AI 辅助代码审查"),
    # 注意：ARTICLES 必须按 stage 聚类排列。
    # build_full() 依赖 `stage != current_stage` 插入阶段标题，
    # 同一 stage 若被拆散会导致阶段标题重复出现。
    ("横切", "16-Skills与Automation.md", "Skills 与 Automation"),
    ("横切", "17-权限管控.md", "权限管控：deny/allow/ask"),
    ("横切", "20-AI生成代码的可解释性债.md", "AI 生成代码的可解释性债"),
    ("交付运维", "18-CI-CD集成.md", "AI 编码进入 CI 流水线"),
    ("交付运维", "19-智能运维异常自愈.md", "智能运维异常自愈"),
    ("交付运维", "22-AI编码自动化水平与失控防线.md", "AI 编码自动化水平与失控防线"),
    ("交付运维", "23-免审的代价：AI编码里省掉的每一次人眼审查都要有东西兑换.md",
     "免审的代价"),
]


def read_article(filepath: Path) -> str:
    """读取文章内容，去掉文件头的 --- YAML front matter ---"""
    text = filepath.read_text(encoding="utf-8")
    # 去除 YAML front matter（如果存在）
    text = re.sub(r'^---\n.*?---\n', '', text, count=1, flags=re.DOTALL)
    return text.strip()


def build_toc() -> str:
    """生成目录"""
    lines = ["# 目录", ""]
    for i, (stage, filename, title) in enumerate(ARTICLES, 1):
        lines.append(f"{i}. [{title}](#{anchor(title)})")
    lines.extend(["", "---", ""])
    return "\n".join(lines)


def anchor(title: str) -> str:
    """生成 GitHub 风格的锚点"""
    return title.lower().replace(" ", "-").replace("：", "").replace(":", "").replace("、", "").replace("/", "").replace("（", "").replace("）", "").replace("，", "")


def build_full() -> str:
    sections = [
        "# AI 编码实践系列 · 完整合辑",
        "",
        f"> 按 SDLC 自然流程组织的 {len(ARTICLES)} 篇方法论文章，来自 Shop-Agent 项目的实战经验。",
        "> 本文档由 `apps/shop-agent/scripts/train/sft/build_full_series.py` 自动生成。",
        "",
        "---",
        "",
        build_toc(),
    ]

    current_stage = None
    for stage, filename, title in ARTICLES:
        filepath = SERIES_DIR / filename
        if not filepath.exists():
            sections.append(f"> [WARN] 文件不存在: {filename}")
            sections.append("")
            continue

        content = read_article(filepath)

        # 阶段标题
        if stage != current_stage:
            current_stage = stage
            sections.extend(["", "---", "", f"# {stage}阶段", "", "---", ""])

        # 文章内容
        sections.append(content)
        sections.extend(["", ""])

    return "\n".join(sections)


def main():
    print(f"读取 {SERIES_DIR}")
    print(f"共 {len(ARTICLES)} 篇文章\n")

    missing = [f for _, f, _ in ARTICLES if not (SERIES_DIR / f).exists()]
    if missing:
        print(f"[WARN] 缺少 {len(missing)} 个文件:")
        for f in missing:
            print(f"  - {f}")
        print()

    full_text = build_full()
    OUTPUT_FILE.write_text(full_text, encoding="utf-8")
    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"[OK] 已生成: {OUTPUT_FILE} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
