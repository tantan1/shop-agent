"""
Demo: drive the Langfuse MLOps annotation flow with representative test data.

This script uses the SAME langfuse v4.7.0 API shapes as
src/modules/monitoring/langfuse_mlops.py (the production code path):
  - trace = root span created via client.start_as_current_observation(as_type="span")
  - trace_id  = client.get_current_trace_id()
  - scores    = client.create_score(trace_id=, name=, value=, data_type="CATEGORICAL")
  - export    = client.api.trace.list(...) + client.api.trace.get(...) -> dataset items

It injects a handful of labelled "tool_select_review" traces covering the Design-4
correction types (correct selection, demo correction, cross-turn rephrase,
clarification reject, approval reject), then builds a Langfuse dataset
"shop-agent-tool-select" from the labelled traces.

Run:  python scripts/demo_annotation_flow.py
"""
from __future__ import annotations

import os
import time
import uuid

from langfuse import Langfuse

HOST = os.environ.get("LANGFUSE_HOST", "http://localhost:3100")
PUBLIC = os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-fd59f364-8eb5-4479-b0cb-034ececdd792")
SECRET = os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-cc521630-c16d-4587-9e52-b518f23ff925")
TRACE_NAME = "tool_select_review"
DATASET_NAME = "shop-agent-tool-select"


def make_client() -> Langfuse:
    return Langfuse(public_key=PUBLIC, secret_key=SECRET, host=HOST)


def create_trace(client: Langfuse, category: str, sample_content: str,
                 session_id: str, conversation_id: str) -> str:
    """Mirror langfuse_mlops._create_trace: root span -> real trace_id."""
    with client.start_as_current_observation(
        as_type="span",
        name=TRACE_NAME,
        metadata={
            "category": category,
            "sample_content": sample_content,
            "session_id": session_id,
            "conversation_id": conversation_id,
        },
        input={"message": sample_content},
    ) as _obs:
        return client.get_current_trace_id()


def write_score(client: Langfuse, trace_id: str, name: str, value) -> None:
    """Mirror langfuse_mlops._write_score (v4 API shape)."""
    client.create_score(
        trace_id=trace_id,
        name=name,
        value=value,
        data_type="CATEGORICAL",
    )


def inject_samples(client: Langfuse) -> list[dict]:
    """Return the list of labelled samples (also used for export)."""
    session = f"demo-session-{uuid.uuid4().hex[:8]}"
    samples = [
        # 1) correct auto selection (clean positive)
        dict(category="refund", sample_content="我要申请退款",
             review_label="correct", correct_tool="refund",
             original_tool="refund", selection_source="auto"),
        # 2) demo correction button (C3 manual)
        dict(category="refund", sample_content="帮我退掉这笔订单",
             review_label="correction", correct_tool="refund",
             original_tool="order_query", selection_source="demo"),
        # 3) cross-turn rephrase positive (C1)
        dict(category="logistics", sample_content="我的快递到哪了",
             review_label="correction", correct_tool="logistics_query",
             original_tool="order_query", selection_source="cross_turn_rephrase"),
        # 4) clarification rejected (C2 A2A input-required)
        dict(category="unknown", sample_content="那个东西怎么弄",
             review_label="correction", correct_tool="clarify",
             original_tool="input_required", selection_source="clarify_reject"),
        # 5) approval rejected (C3 risky tool)
        dict(category="coupon", sample_content="给我发一张大额优惠券",
             review_label="correction", correct_tool="coupon_apply",
             original_tool="coupon_grant", selection_source="approve_reject"),
        # 6) another clean positive for dataset diversity
        dict(category="order", sample_content="查一下我的订单状态",
             review_label="correct", correct_tool="order_query",
             original_tool="order_query", selection_source="auto"),
    ]

    created = []
    for i, s in enumerate(samples):
        conv = f"{session}-c{i}"
        tid = create_trace(client, s["category"], s["sample_content"], session, conv)
        write_score(client, tid, "review_label", s["review_label"])
        write_score(client, tid, "correct_tool", s["correct_tool"])
        write_score(client, tid, "selection_source", s["selection_source"])
        if s["review_label"] == "correction":
            write_score(client, tid, "original_tool", s["original_tool"])
        created.append({**s, "trace_id": tid, "conversation_id": conv})
        print(f"  [{i + 1}] trace={tid[:8]}.. label={s['review_label']:9} "
              f"tool={s['correct_tool']:14} src={s['selection_source']}")
    return created


def export_to_dataset(client: Langfuse, created: list[dict]) -> int:
    """Mirror langfuse_mlops.export_and_train: pull labelled traces -> dataset."""
    # v4 requires the dataset to exist before items can be appended
    try:
        client.create_dataset(name=DATASET_NAME)
    except Exception:
        pass  # already exists

    # collect source trace ids already present (idempotent re-runs)
    existing = set()
    try:
        ds = client.get_dataset(name=DATASET_NAME)
        for it in getattr(ds, "items", []) or []:
            st = getattr(it, "source_trace_id", None)
            if st:
                existing.add(st)
    except Exception:
        pass

    page = client.api.trace.list(name=TRACE_NAME, limit=50)
    traces = getattr(page, "data", page) or []
    print(f"  pulled {len(traces)} '{TRACE_NAME}' trace(s) from Langfuse")

    n = 0
    for t in traces:
        tid = t.id if hasattr(t, "id") else t.get("id")
        if tid in existing:
            continue
        full = client.api.trace.get(trace_id=tid)
        scores = getattr(full, "scores", None) or []
        # v4 categorical scores carry the value in .string_value (numeric .value is 0.0)
        def _score(name):
            for s in scores:
                if getattr(s, "name", "") == name:
                    return getattr(s, "string_value", None) or getattr(s, "value", None)
            return None

        label = _score("review_label")
        correct_tool = _score("correct_tool")
        src = _score("selection_source")
        if label is None or correct_tool is None:
            continue
        # idempotent dataset creation keyed by source trace id
        # (matches langfuse_mlops._maybe_create_dataset v4 call shape)
        try:
            client.create_dataset_item(
                dataset_name=DATASET_NAME,
                input={"query": getattr(full, "input", {}).get("message") if isinstance(getattr(full, "input", None), dict) else getattr(full, "input", None)},
                expected_output=correct_tool,
                metadata={"selection_source": src, "category": getattr(full, "metadata", {}).get("category"), "review_label": label},
                source_trace_id=tid,
            )
            n += 1
        except Exception as e:  # duplicate source_trace_id etc.
            print(f"  skip dataset item for {tid[:8]}.. ({e})")
    return n


def main() -> None:
    print(f"Langfuse host = {HOST}")
    client = make_client()
    print("Injecting representative annotation samples into Langfuse ...")
    created = inject_samples(client)
    print("Flushing ingestion events ...")
    client.flush()
    time.sleep(10)  # allow worker to persist scores onto traces
    print("Exporting labelled traces into dataset "
          f"'{DATASET_NAME}' ...")
    n = export_to_dataset(client, created)
    client.flush()
    print(f"Done. {len(created)} traces created, {n} dataset items exported.")
    print(f"Open Langfuse UI -> Traces (name='{TRACE_NAME}') and Datasets -> '{DATASET_NAME}'")
    print(f"UI: {HOST}")


if __name__ == "__main__":
    main()
