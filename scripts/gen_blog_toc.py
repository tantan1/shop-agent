#!/usr/bin/env python3
"""
提取 shop-agent-blog-series 所有带编号博文的标题结构，生成目录大纲。
用法:
  python scripts/gen_blog_toc.py          # 本地版（带相对链接）
  python scripts/gen_blog_toc.py --csdn   # CSDN 版（纯文本，无链接）
输出:
  docs/shop-agent-blog-series/目录大纲.md
  docs/shop-agent-blog-series/目录大纲-csdn.md  (--csdn 时)
"""

import re
import sys
from pathlib import Path

BLOG_DIR = Path(__file__).resolve().parent.parent / "docs" / "shop-agent-blog-series"
OUTPUT_FILE = BLOG_DIR / "目录大纲.md"
OUTPUT_FILE_CSDN = BLOG_DIR / "目录大纲-csdn.md"


def extract_headings(text: str):
    headings = []
    for line in text.splitlines():
        m = re.match(r"^(#{1,2})\s+(.+)$", line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            headings.append((level, title))
    return headings


def slugify(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[\s\-—:：，。！？、；（）()\[\]【】<>]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    text = text.strip("-")
    return text or "section"


def guess_title_from_filename(fname: str) -> str:
    stem = Path(fname).stem
    stem = re.sub(r"^\d+\-", "", stem)
    stem = stem.replace("-", " ")
    return stem


def generate_local(files):
    lines = []
    lines.append("# 系列目录大纲\n")
    lines.append("> 自动生成，基于各篇 H1/H2 标题。\n")
    lines.append("---\n")

    for f in files:
        text = f.read_text(encoding="utf-8")
        headings = extract_headings(text)

        rel_link = f.name

        if headings and headings[0][0] == 1:
            h1_title = headings[0][1]
        elif headings:
            h1_title = headings[0][1]
        else:
            h1_title = guess_title_from_filename(f.name)

        lines.append(f"## [{h1_title}]({rel_link})\n")

        for level, title in headings:
            if level == 1:
                continue
            indent = "  " * (level - 2)
            anchor = slugify(title)
            lines.append(f"{indent}- [{title}]({rel_link}#{anchor})\n")

        lines.append("\n")

    return "".join(lines)


def generate_csdn(files):
    lines = []
    lines.append("# 系列目录大纲\n")
    lines.append("> 自动生成，基于各篇 H1/H2 标题。\n")
    lines.append("---\n")

    for f in files:
        text = f.read_text(encoding="utf-8")
        headings = extract_headings(text)

        if headings and headings[0][0] == 1:
            h1_title = headings[0][1]
        elif headings:
            h1_title = headings[0][1]
        else:
            h1_title = guess_title_from_filename(f.name)

        lines.append(f"## {h1_title}\n")

        for level, title in headings:
            if level == 1:
                continue
            indent = "  " * (level - 2)
            lines.append(f"{indent}- {title}\n")

        lines.append("\n")

    return "".join(lines)


def main():
    csdn_mode = "--csdn" in sys.argv

    files = sorted(
        p for p in BLOG_DIR.glob("*-*.md") if p.is_file()
    )

    if csdn_mode:
        content = generate_csdn(files)
        OUTPUT_FILE_CSDN.write_text(content, encoding="utf-8")
        print(f"已生成 CSDN 版: {OUTPUT_FILE_CSDN}")
    else:
        content = generate_local(files)
        OUTPUT_FILE.write_text(content, encoding="utf-8")
        print(f"已生成本地版: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
