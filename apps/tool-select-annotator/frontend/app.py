"""标注 Web 薄层前端(Streamlit,§2.2)。

运行: streamlit run frontend/app.py  (后端 BASE_URL 通过 env 或下方常量配置)
"""
import os
import requests
import streamlit as st

BASE_URL = os.environ.get("ANNOTATOR_BASE_URL", "http://localhost:8137")
PAGE_SIZE = int(os.environ.get("PAGE_SIZE", "20"))


def _get(path, params=None):
    return requests.get(f"{BASE_URL}{path}", params=params, timeout=15).json()


def _post(path, payload):
    return requests.post(f"{BASE_URL}{path}", json=payload, timeout=15).json()


st.set_page_config(page_title="工具选择标注台", layout="wide")
st.title("🛠️ 工具选择标注台 (tool-select-annotator)")

tab = st.sidebar.radio("视图", ["人工队列", "统计"])

if tab == "统计":
    s = _get("/stats")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总量", s["total"])
    col2.metric("待人工", s["unlabeled_in_queue"])
    col3.metric("自动银标", s["auto_labeled"])
    col4.metric("人工已标", s["manual_labeled"])
    st.metric("抽样复核(spotcheck)", s["spotcheck"])
    st.metric("用户反馈信号", s["feedback_signals"])
    st.metric("修订次数(Relabel)", s["relabel_count"])
    st.caption("自动银标占比高 = 复检减负有效;spotcheck 一致率低时需收紧 AUTO_PASS_MARGIN。")
else:
    cat = st.sidebar.text_input("category 过滤(可选)")
    if "page" not in st.session_state:
        st.session_state.page = 0

    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("⬅ 上一页"):
        st.session_state.page = max(0, st.session_state.page - 1)
    if c2.button("下一页 ➡"):
        st.session_state.page += 1
    if c3.button("🔄 增量同步 Langfuse"):
        res = _post("/sync", {})
        st.toast(f"同步: 拉取 {res.get('pulled')} / 自动银标 {res.get('auto_labeled')} / "
                 f"spotcheck {res.get('spotcheck')}" +
                 (f" | 错误: {res.get('error')}" if res.get("error") else ""))

    rows = _get("/candidates", {"category": cat or None, "page": st.session_state.page,
                                "page_size": PAGE_SIZE})
    if not rows:
        st.info("队列为空:可点『增量同步』拉取新 trace,或已全部标注。")
    # 工具清单(名称+说明),供候选选项与描述展示(§2.1)
    if "catalog" not in st.session_state:
        st.session_state.catalog = _get("/tools").get("tools", [])
    catalog = st.session_state.catalog
    desc_map = {t["name"]: (t.get("description") or "") for t in catalog}

    for r in rows:
        with st.expander(f"[{r['category']}] {r['query_text'][:60]}  "
                         f"(freq={r['freq']}, margin={r['margin']}, conf={r['label_source']})"):
            st.write("**query:**", r["query_text"])
            st.write("**多意图:**", r["is_multi_intent"])
            sel = r["top1_tool"] or r["llm_suggested_tool"]
            if sel:
                st.info(f"🤖 模型选择: **{sel}**  (margin={r['margin']})")
            else:
                st.caption("（该 trace 无模型选择记录）")

            # 候选工具集:trace 自带 available_tools 优先;历史 trace 无则回退全局清单
            available = r["available_tools"] or []
            tool_options = available if available else [t["name"] for t in catalog]
            if tool_options:
                st.markdown("**可选工具及说明:**")
                for nm in tool_options:
                    st.caption(f"• `{nm}` — {desc_map.get(nm, '')}")
            else:
                st.warning("该 trace 无候选工具列表,且工具清单不可用,请自由填写。")

            default = sel or ""
            multi = r["is_multi_intent"]

            if multi:
                st.caption("🔀 多意图:可勾选多个正确工具")
                if tool_options:
                    correct_tools = st.multiselect(
                        "正确工具(多选) gold_tools",
                        options=tool_options,
                        default=[default] if default in tool_options else [],
                        key=f"gold_{r['canonical_tid']}",
                    )
                else:
                    raw = st.text_input(
                        "正确工具(逗号分隔,多意图可填多个)",
                        value=default, key=f"gold_{r['canonical_tid']}",
                    )
                    correct_tools = [t.strip() for t in raw.split(",") if t.strip()]
                correct_tool = correct_tools[0] if correct_tools else None
            else:
                if tool_options:
                    correct_tool = st.selectbox(
                        "正确工具 gold_tool",
                        options=[""] + tool_options,
                        index=(tool_options.index(default) + 1) if default in tool_options else 0,
                        key=f"gold_{r['canonical_tid']}",
                    )
                else:
                    correct_tool = st.text_input(
                        "正确工具(自由填写)",
                        value=default, key=f"gold_{r['canonical_tid']}",
                    ) or None
                correct_tools = [correct_tool] if correct_tool else []

            rejected = st.multiselect(
                "被否定工具 rejected_tools",
                options=tool_options,
                key=f"rej_{r['canonical_tid']}",
            )
            # 自由补刀:历史 trace 候选为空时,允许额外加被否定工具
            if not tool_options:
                rej_raw = st.text_input("被否定工具(逗号分隔,可选)", key=f"rejraw_{r['canonical_tid']}")
                if rej_raw:
                    rejected = rejected + [t.strip() for t in rej_raw.split(",") if t.strip()]

            reason = st.text_input("修订原因(可选)", key=f"reason_{r['canonical_tid']}")
            if st.button("提交标注", key=f"sub_{r['canonical_tid']}"):
                res = _post("/submit/gold", {
                    "canonical_tid": r["canonical_tid"],
                    "correct_tool": correct_tool or None,
                    "correct_tools": correct_tools or None,
                    "rejected_tools": rejected or None,
                    "reason": reason or None,
                    "label_source": "confirm" if (correct_tool == default and len(correct_tools) == 1) else "full",
                })
                if res.get("ok"):
                    st.success("已提交" + ("(Relabel 修订已记审计)" if res.get("relabeled") else ""))
                    st.rerun()
                else:
                    st.error(res.get("error"))

            # 用户反馈(§2.5)
            fcol1, fcol2 = st.columns(2)
            if fcol1.button("👍 点赞", key=f"like_{r['canonical_tid']}"):
                _post("/submit/feedback", {"trace_id": r["trace_id"],
                                           "conversation_id": r["conversation_id"], "signal": "like"})
                st.toast("已记录点赞")
            if fcol2.button("👎 点踩", key=f"dislike_{r['canonical_tid']}"):
                _post("/submit/feedback", {"trace_id": r["trace_id"],
                                           "conversation_id": r["conversation_id"], "signal": "dislike"})
                st.toast("已记录点踩")
