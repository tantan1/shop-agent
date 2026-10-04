"""工具选择三方案对比 - Arm A(余弦) / Arm B(PyTorch 线性头)基线评测。

定位：本脚本是「你写、AI 辅助」模式下的脚手架。
  - 管线 / 数据加载 / per_case 输出（非学习部分）已写好，可直接跑 Arm A。
  - Arm B 的 PyTorch 核心（T4 Dataset / T5 Model / T6 训练循环 / T7 超参搜索）
    留成 TODO 缺口，由你来实现——这正是 26-PyTorch学习路径.md 的实操练习。

数据一致性（计划文档 §0）：三 arm 共用同一个 data/lscale_*.json 的
  tools[]（描述，作 embedding 原型）与 queries[]（message + correct_tool 金标）。

输出段名（对齐 compare_tool_selection.py::extract_per_case）：
  Arm A -> "p0p1"  : per_case 含 p1_top1 / p1_top1_hit
  Arm B -> "fc"    : per_case 含 selected / fc_hit
  Arm C -> "p2"    : 由现有 eval_tool_select_sft.py 产出

模型版本一致性：
  - 训练时记录 embedding 模型的版本标识（本地文件哈希或 HF 版本）
  - 推理/部署加载 head 时自动比对，不一致则报错提示重训
  - 保证「训练用的 embedding」与「推理用的 embedding」完全一致

用法：
  # Arm A：先跑通整条 pipeline（无需任何训练）
  .\\venv_cuda\\Scripts\\python scripts/eval/tool_select/eval_embedding_baseline.py \\
      --arm a --data data/lscale_S.json --out outputs/armA_S.json

  # Arm B：先训练（你填完 T4~T6 后）
  .\\venv_cuda\\Scripts\\python scripts/eval/tool_select/eval_embedding_baseline.py \\
      --arm b --train --train-data data/llamafactory/shop_tool_select_S.json \\
      --eval-data data/lscale_S.json --out outputs/armB_S.json
"""
import argparse
import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # 训练/推理统一走远程 endpoint 时可不安装 sentence-transformers
    SentenceTransformer = None
from torch.utils.data import Dataset, DataLoader

from torch.utils.data import Dataset, DataLoader


# ================================================================
# 模型版本一致性工具
# ================================================================
def _model_local_path(model_name: str) -> Path | None:
    """返回本地模型目录路径（models/BAAI/xxx），不存在返回 None。"""
    # 优先找 repo_root/models/BAAI/xxx
    repo_root = _find_repo_root()
    local = repo_root / "models" / "BAAI" / model_name.split("/")[-1]
    if local.exists():
        return local
    # 兼容 HF cache
    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    for cand in hf_cache.glob(f"hub/models--{model_name.replace('/', '--')}*/snapshots/*"):
        if cand.is_dir():
            return cand
    return None


def compute_model_version(model_name: str) -> str:
    """计算 embedding 模型的版本标识。
    优先用本地文件的 MD5（前 8 位），回退到 model_name。
    """
    local = _model_local_path(model_name)
    if local and local.is_dir():
        # 取前几个大文件的哈希组合，避免全目录遍历太慢
        hashes = []
        for f in sorted(local.rglob("*")):
            if f.is_file() and f.suffix in {".bin", ".safetensors", ".json", ".txt", ".model"}:
                try:
                    h = hashlib.md5(f.read_bytes()).hexdigest()[:8]
                    hashes.append(h)
                    if len(hashes) >= 5:
                        break
                except Exception:
                    pass
        if hashes:
            combined = "".join(sorted(hashes))
            return hashlib.md5(combined.encode()).hexdigest()[:12]
    # 回退：用模型名的哈希
    return hashlib.md5(model_name.encode()).hexdigest()[:12]


def save_head_with_version(model_path: str, embed_model: str, embed_version: str, **extra_meta):
    """保存 head 权重时同时写入版本元数据（同目录下 .meta.json）。"""
    meta = {
        "embed_model": embed_model,
        "embed_version": embed_version,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra_meta,
    }
    meta_path = Path(model_path).with_suffix(".meta.json")
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[INFO] 元数据已写入 {meta_path}")


def load_head_with_version_check(model_path: str, current_embed_model: str, current_embed_version: str):
    """加载 head 时比对 embedding 版本，不一致抛出异常提示重训。"""
    meta_path = Path(model_path).with_suffix(".meta.json")
    if not meta_path.exists():
        raise RuntimeError(
            f"未找到模型元数据 {meta_path}，无法验证 embedding 版本一致性。"
            f"请重新训练（--train）生成新权重。"
        )
    meta = json.load(open(meta_path, encoding="utf-8"))
    saved_model = meta.get("embed_model")
    saved_version = meta.get("embed_version")
    if saved_model != current_embed_model or saved_version != current_embed_version:
        raise RuntimeError(
            f"Embedding 模型版本不匹配，必须重训：\n"
            f"  训练时: {saved_model} (version={saved_version})\n"
            f"  当前  : {current_embed_model} (version={current_embed_version})\n"
            f"请执行：--train 重新训练。"
        )
    print(f"[OK] Embedding 版本一致性通过: {current_embed_model} v{current_embed_version}")


# ----------------------------------------------------------------
# 路径：向上找到含 data/lscale_S.json 的仓库根
# ----------------------------------------------------------------
def _find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for cand in [p, *p.parents]:
        if (cand / "data" / "lscale_S.json").exists():
            return cand
    raise FileNotFoundError("找不到 data/lscale_S.json，请在 --data 显式指定")


REPO_ROOT = _find_repo_root()
DEFAULT_EMBED = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")


class EndpointEmbedder:
    """OpenAI 兼容 /v1/embeddings 客户端，接口对齐 SentenceTransformer.encode。

    作用：训练/部署/推理统一到同一个 embedding 服务——训练时从这里取向量，
    部署时 ToolHeadClassifier 也调同一个端点，保证训练与推理 embedding 完全一致
    （避免「训练用 bge-small-zh 本地模型、推理用 bge-small-zh vLLM」的向量错位）。
    """

    def __init__(self, base_url: str, model: str = "BAAI/bge-small-zh-v1.5"):
        self.url = base_url.rstrip("/") + "/v1/embeddings"
        self.model = model

    def encode(self, texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        import httpx

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = httpx.post(self.url, json={"model": self.model, "input": batch}, timeout=60)
            resp.raise_for_status()
            items = sorted(resp.json()["data"], key=lambda it: it.get("index", 0))
            out.extend(it["embedding"] for it in items)
        arr = np.array(out, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr


# ================================================================
# ① 数据加载（plumbing，已写好）
# ================================================================
def load_lscale(path: str):
    """返回 (tools, queries)。tools=[{name,description,...}]，queries=[{message,correct_tool,level}]。"""
    ds = json.load(open(path, encoding="utf-8"))
    return ds["tools"], ds["queries"]


def tool_desc_text(t: dict) -> str:
    """与 benchmark_tool_selection_pipeline.py 一致的工具描述文本格式。"""
    return f"工具名称：{t['name']}；功能描述：{t.get('description', '')}"


def build_tool_matrix(model: SentenceTransformer, tools: list) -> tuple:
    """预计算工具原型矩阵。返回 (names:list, matrix:Tensor[N,dim])。归一化后内积=余弦。"""
    texts = [tool_desc_text(t) for t in tools]
    emb = model.encode(texts, normalize_embeddings=True)
    names = [t["name"] for t in tools]
    return names, torch.tensor(emb, dtype=torch.float32)


# ================================================================
# ② Arm A 余弦相似度（无训练，已写好，直接可跑）
# ================================================================
def run_arm_a(model, tool_names, tool_matrix, queries, top_k=1, M=0, distract_seed=20260728, pool_by_name=None):
    per_case = []
    rng = random.Random(distract_seed) if M and M > 0 else None
    for idx, q in enumerate(queries):
        msg = q["message"]
        gold = q["correct_tool"]
        t0 = time.perf_counter()
        if M and M > 0 and pool_by_name:
            pool_names_list = list(pool_by_name.keys())
            others = [t for t in pool_names_list if t != gold]
            distract = rng.sample(others, min(M - 1, len(others)))
            cands = [gold] + distract
            cand_names = cands
            cand_matrix = torch.stack([tool_matrix[pool_by_name[n]] for n in cands])
        else:
            cand_names = tool_names
            cand_matrix = tool_matrix
        q_emb = torch.tensor(
            model.encode([msg], normalize_embeddings=True)[0], dtype=torch.float32
        )
        sims = torch.matmul(cand_matrix, q_emb)
        top = torch.topk(sims, k=min(top_k, len(cand_names)))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        best_name = cand_names[top.indices[0].item()]
        hit = best_name == gold
        per_case.append({
            "idx": idx,
            "message": msg,
            "correct_tool": gold,
            "level": q.get("level", "exact"),
            "p1_top1": best_name,
            "p1_top1_hit": bool(hit),
            "latency_ms": round(latency_ms, 3),
            "scores": {cand_names[i]: round(sims[i].item(), 4) for i in top.indices.tolist()},
        })
    return {"p0p1": {"per_case": per_case, "mode": "p0p1"}}


# ================================================================
# ③ Arm B 数据准备（T3，已写好：抽对 + 去重防泄漏）
#    注意：去重是强制的——SFT 训练集由 lscale 增广而来，若不剔除与
#    测试集重叠的 query，Arm B 会"背答案"虚高。
# ================================================================
def extract_train_pairs(sft_path: str, test_msgs: set) -> list:
    """从 shop_tool_select_*.json 抽 (query_text, correct_tool)，并按 message 剔除测试集重叠项。"""
    data = json.load(open(sft_path, encoding="utf-8"))
    pairs = []
    for item in data:
        # SFT 数据：conversations[role==user].content 为 query；顶层 correct_tool 为金标
        user_turn = next(c for c in item["conversations"] if c["role"] == "user")
        msg = user_turn["content"]
        if msg in test_msgs:          # 去重防泄漏
            continue
        pairs.append((msg, item["correct_tool"]))
    return pairs


def build_label_index(tool_names: list):
    return {n: i for i, n in enumerate(tool_names)}


# ================================================================
# ④ Arm B —— PyTorch 学习核心（T4~T7，由你来实现）
#    下面只给签名与脚手架，核心几行用 raise 占位，逼你亲手写。
#    提示见各函数 docstring，对应 26-PyTorch学习路径.md 的学习点。
# ================================================================

# ---- T4：Dataset / DataLoader ----
class ToolSelectDataset(Dataset):
    """__getitem__ 返回 (query_embed:Tensor[dim], label:int)。
    学习点：Dataset 协议 = __len__ + __getitem__；DataLoader 负责 batch/shuffle/collate。
    """

    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, i):
        return self.embeddings[i], self.labels[i]


# ---- T5：nn.Module 定义（线性 / MLP 可切换） ----
class ToolHead(torch.nn.Module):
    """学习点 26-§'nn.Module 定义'：参数与维度映射。
    hidden==0 -> 纯线性 nn.Linear(dim, n_classes)
    hidden>0  -> Linear(dim,hidden) -> ReLU -> Dropout -> Linear(hidden,n_classes)
    """

    def __init__(self, dim: int, n_classes: int, hidden: int = 0, dropout: float = 0.0):
        super().__init__()
        if hidden == 0:
            self.linear = torch.nn.Linear(dim, n_classes)
        else:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(dim, hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, n_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "linear"):
            return self.linear(x)
        return self.mlp(x)


# ---- T6：训练循环（学习重点：forward/loss/backward/step/zero_grad） ----
def train_head(model, train_loader, val_loader, lr: float, epochs: int, device: str):
    """学习点 26-§'训练循环从 YAML 改成原生 for 循环' + '混合精度 autocast'。
    返回训练好的 model，并打印每 epoch 的 train_loss / val_acc。
    """
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            if device.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
            train_loss += loss.item() * xb.size(0)
            train_correct += (logits.argmax(dim=-1) == yb).sum().item()
            train_total += xb.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (logits.argmax(dim=-1) == yb).sum().item()
                val_total += xb.size(0)

        tr_loss = train_loss / train_total if train_total else 0.0
        tr_acc = train_correct / train_total if train_total else 0.0
        vl_loss = val_loss / val_total if val_total else 0.0
        vl_acc = val_correct / val_total if val_total else 0.0
        print(
            f"Epoch {ep+1:>3d}/{epochs}  "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
            f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f}"
        )
    return model


# ---- T7：超参搜索（学习重点：参数调整） ----
def hp_search(train_loader, val_loader, dim, n_classes, device):
    """学习点 26-§'超参数调优'：网格搜索 lr/hidden/epochs/dropout，按 val_acc 选最优。
    返回 (best_model, best_cfg)。
    """
    import itertools

    lrs = [1e-3, 1e-2, 1e-1]
    hiddens = [0, 64, 128]
    epochs_list = [20, 50]
    dropouts = [0.0, 0.2]

    best_acc = -1.0
    best_cfg = None
    best_model = None

    for lr, hidden, epochs, dropout in itertools.product(lrs, hiddens, epochs_list, dropouts):
        model = ToolHead(dim, n_classes, hidden=hidden, dropout=dropout)
        model = model.to(device)
        train_head(model, train_loader, val_loader, lr=lr, epochs=epochs, device=device)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                correct += (logits.argmax(dim=-1) == yb).sum().item()
                total += yb.size(0)
        acc = correct / total if total else 0.0
        print(f"[HP] lr={lr}, hidden={hidden}, epochs={epochs}, dropout={dropout} -> val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_cfg = {"lr": lr, "hidden": hidden, "epochs": epochs, "dropout": dropout}
            best_model = model

    return best_model, best_cfg


# ================================================================
# ⑤ Arm B 推理（T8，待你训完模型后接入；此处仅留接口）
# ================================================================
def run_arm_b(model, tool_names, queries, top_k=1, device="cpu", M=0, distract_seed=20260728, pool_by_name=None):
    per_case = []
    model.eval()
    rng = random.Random(distract_seed) if M and M > 0 else None
    for idx, q in enumerate(queries):
        msg = q["message"]
        gold = q["correct_tool"]
        t0 = time.perf_counter()
        with torch.no_grad():
            q_emb = torch.tensor(
                _EMBED_MODEL.encode([msg], normalize_embeddings=True)[0],
                dtype=torch.float32,
            ).unsqueeze(0).to(device)
            logits = model(q_emb)
            if M and M > 0 and pool_by_name:
                pool_names_list = list(pool_by_name.keys())
                others = [t for t in pool_names_list if t != gold]
                distract = rng.sample(others, min(M - 1, len(others)))
                cands = [gold] + distract
                cand_logits = torch.tensor([logits[0, tool_names.index(n)].item() for n in cands], device=device)
                probs = torch.softmax(cand_logits, dim=-1)
                top = torch.topk(probs, k=min(top_k, len(cands)))
                best = cands[top.indices[0].item()]
                score_dict = {cands[i]: round(probs[i].item(), 4) for i in top.indices.tolist()}
            else:
                probs = torch.softmax(logits, dim=-1)[0]
                top = torch.topk(probs, k=min(top_k, len(tool_names)))
                best = tool_names[top.indices[0].item()]
                score_dict = {tool_names[i]: round(probs[i].item(), 4) for i in top.indices.tolist()}
        latency_ms = (time.perf_counter() - t0) * 1000.0
        per_case.append({
            "idx": idx,
            "message": msg,
            "correct_tool": gold,
            "level": q.get("level", "exact"),
            "selected": best,
            "fc_hit": bool(best == gold),
            "latency_ms": round(latency_ms, 3),
            "scores": score_dict,
        })
    return {"fc": {"per_case": per_case, "mode": "fc",
                   "p50_latency_ms": round(statistics.median([pc["latency_ms"] for pc in per_case]), 1) if per_case else None,
                   "p95_latency_ms": round(_pctl([pc["latency_ms"] for pc in per_case], 0.95), 1) if per_case else None}}


def _pctl(values: list, p: float) -> float:
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def infer_hidden_from_state(model_out: str, dim: int, n_classes: int) -> ToolHead:
    state = torch.load(model_out, map_location="cpu")
    if "linear.weight" in state:
        model = ToolHead(dim, n_classes, hidden=0)
    elif "mlp.0.weight" in state:
        hidden = state["mlp.0.weight"].shape[0]
        model = ToolHead(dim, n_classes, hidden=hidden)
    else:
        raise KeyError("无法从 state_dict 推断模型结构")
    return model


# ================================================================
# ⑥ 入口
# ================================================================
def main():
    global _EMBED_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["a", "b"], required=True)
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "lscale_S.json"))
    ap.add_argument("--out", default="outputs/arm_out.json")
    ap.add_argument("--embed-model", default=DEFAULT_EMBED)
    ap.add_argument("--embed-endpoint", default=None,
                    help="OpenAI 兼容 /v1/embeddings 端点（如 http://localhost:18080）。"
                         "指定后训练/推理统一走该端点，不再本地加载模型（与部署共用同一 embedding）。")
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--M", type=int, default=0, help="候选集大小：0=全集；>0 时从工具池随机抽 M-1 干扰+gold")
    ap.add_argument("--distract-seed", type=int, default=20260728, help="干扰项抽样种子")
    # Arm B 专用
    ap.add_argument("--train", action="store_true", help="训练 Arm B（需先实现 T4~T6）")
    ap.add_argument("--train-data", default=str(REPO_ROOT / "data" / "llamafactory" / "shop_tool_select_S.json"))
    ap.add_argument("--model-out", default="outputs/tool_head_prod.pt", help="训练后权重保存路径")
    ap.add_argument("--epochs", type=int, default=0, help="直接训练轮数；0=使用 hp_search")
    ap.add_argument("--lr", type=float, default=1e-3, help="直接训练学习率（--epochs>0 时生效）")
    ap.add_argument("--hidden", type=int, default=64, help="直接训练隐层维度（--epochs>0 时生效）")
    ap.add_argument("--dropout", type=float, default=0.0, help="直接训练 dropout（--epochs>0 时生效）")
    args = ap.parse_args()

    if args.embed_endpoint:
        print(f"[INFO] embedding endpoint: {args.embed_endpoint} model={args.embed_model}")
        _EMBED_MODEL = EndpointEmbedder(args.embed_endpoint, args.embed_model)
        embed_version = hashlib.md5(args.embed_model.encode()).hexdigest()[:12]
    else:
        print(f"[INFO] Loading local embedding model {args.embed_model} ...")
        assert SentenceTransformer is not None, "sentence-transformers not installed and --embed-endpoint not specified"
        local_path = _model_local_path(args.embed_model)
        if local_path is None:
            raise RuntimeError(f"Local embedding model not found: {args.embed_model}, please place at E:/workspace/shop-agent/models/BAAI/bge-small-zh-v1.5")
        print(f"[INFO] Using local model: {local_path}")
        _EMBED_MODEL = SentenceTransformer(str(local_path))
        embed_version = compute_model_version(args.embed_model)
    print(f"[INFO] Embedding model: {args.embed_model} version: {embed_version}")

    tools, queries = load_lscale(args.data)
    tool_names, tool_matrix = build_tool_matrix(_EMBED_MODEL, tools)
    label2idx = build_label_index(tool_names)
    pool_by_name = {t["name"]: i for i, t in enumerate(tools)} if args.M and args.M > 0 else None
    print(f"[INFO] 工具数={len(tool_names)}，query 数={len(queries)}，emb_dim={tool_matrix.shape[1]}, M={args.M}")

    if args.arm == "a":
        summary = run_arm_a(_EMBED_MODEL, tool_names, tool_matrix, queries, top_k=args.top_k,
                            M=args.M, distract_seed=args.distract_seed, pool_by_name=pool_by_name)
        lats = [pc["latency_ms"] for pc in summary["p0p1"]["per_case"] if isinstance(pc.get("latency_ms"), (int, float)) and pc["latency_ms"] > 0]
        if lats:
            summary["p0p1"]["p50_latency_ms"] = round(statistics.median(lats), 1)
            summary["p0p1"]["p95_latency_ms"] = round(_pctl(lats, 0.95), 1)
    else:  # arm b
        if args.train:
            test_msgs = {q["message"] for q in queries}
            pairs = extract_train_pairs(args.train_data, test_msgs)
            print(f"[INFO] 训练对（去重后）={len(pairs)}")
            if len(pairs) == 0:
                raise RuntimeError("训练对为空，请检查训练数据与去重逻辑")

            msgs = [p[0] for p in pairs]
            labels = [label2idx[p[1]] for p in pairs]
            print(f"[INFO] 正在嵌入 {len(msgs)} 条训练 query ...")
            emb = _EMBED_MODEL.encode(msgs, normalize_embeddings=True)
            x = torch.tensor(emb, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.long)

            n = len(x)
            n_val = max(1, int(n * 0.2))
            perm = torch.randperm(n)
            train_idx = perm[n_val:]
            val_idx = perm[:n_val]
            train_ds = ToolSelectDataset(x[train_idx], y[train_idx])
            val_ds = ToolSelectDataset(x[val_idx], y[val_idx])
            train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

            dim = x.shape[1]
            n_classes = len(tool_names)
            device = args.device
            print(f"[INFO] dim={dim}, n_classes={n_classes}, train={len(train_ds)}, val={len(val_ds)}, device={device}")

            if args.epochs > 0:
                model = ToolHead(dim, n_classes, hidden=args.hidden, dropout=args.dropout)
                model = model.to(device)
                train_head(model, train_loader, val_loader, lr=args.lr, epochs=args.epochs, device=device)
                best_cfg = {"lr": args.lr, "hidden": args.hidden, "epochs": args.epochs, "dropout": args.dropout}
            else:
                model, best_cfg = hp_search(train_loader, val_loader, dim, n_classes, device)
            print(f"[INFO] 最优超参: {best_cfg}")

            Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.model_out)
            # 保存 embedding 版本元数据
            save_head_with_version(
                args.model_out,
                embed_model=args.embed_model,
                embed_version=embed_version,
                best_cfg=best_cfg,
                dim=dim,
                n_classes=n_classes,
            )
            print(f"[OK] 模型权重已保存到 {args.model_out}")

            model.load_state_dict(torch.load(args.model_out, map_location=args.device))
            model = model.to(args.device)
            summary = run_arm_b(model, tool_names, queries, top_k=args.top_k, device=args.device,
                                M=args.M, distract_seed=args.distract_seed, pool_by_name=pool_by_name)
        else:
            dim = tool_matrix.shape[1]
            n_classes = len(tool_names)
            # 加载前校验 embedding 版本一致性
            load_head_with_version_check(args.model_out, args.embed_model, embed_version)
            model = infer_hidden_from_state(args.model_out, dim, n_classes)
            model.load_state_dict(torch.load(args.model_out, map_location=args.device))
            model = model.to(args.device)
            summary = run_arm_b(model, tool_names, queries, top_k=args.top_k, device=args.device,
                                M=args.M, distract_seed=args.distract_seed, pool_by_name=pool_by_name)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[OK] 写出 {args.out}")


if __name__ == "__main__":
    main()
