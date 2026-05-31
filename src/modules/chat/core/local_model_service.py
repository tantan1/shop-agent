"""
本地小模型服务（transformers 直接加载，零部署、纯本地）
用于参数抽取等轻量任务，替代 Qwen 云端 API。

依赖: transformers, torch（已在 requirements.txt）
可选: bitsandbytes（4bit 量化，节省内存）
"""

import json
import re
import asyncio
import time as _perf_time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Type, Any
from pydantic import BaseModel

from langfuse import observe

from src.shared.logger import APILogger
from src.modules.chat.config import chat_config

logger = APILogger("local_model")


class LocalModelService:
    """
    本地小模型推理服务

    特性:
    - 纯 transformers 加载，无需 Ollama/vLLM 等额外部署
    - 支持 4bit 量化以节省内存（~1GB）
    - 首次加载自动下载模型（通过 HuggingFace）
    - 专门为参数抽取 JSON 输出优化了 prompt 和解析
    """

    _instance = None
    _model = None
    _tokenizer = None
    _load_failed: bool = False      # True 表示加载失败，不再重试
    _load_failed_reason: str = ""   # 失败原因，供日志输出
    _executor: ThreadPoolExecutor = None  # 推理线程池

    # ── 工具选择器专用模型（可独立配置，与参数抽取模型分开） ──
    _tool_selector_model = None
    _tool_selector_tokenizer = None
    _tool_selector_load_failed: bool = False
    _tool_selector_load_failed_reason: str = ""

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "LocalModelService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_loaded(self) -> bool:
        """懒加载模型（首次调用时加载）。返回 True 表示可用，False 表示不可用。"""
        if self._model is not None:
            return True

        # 曾经加载失败过，不再重试（避免日志刷屏）
        if self._load_failed:
            logger.debug(f"本地模型加载已失败，跳过: {self._load_failed_reason}")
            return False

        model_name = chat_config.local_param_model
        device = chat_config.local_param_device
        use_4bit = chat_config.local_param_load_in_4bit

        logger.info(f"正在加载本地模型: {model_name} (device={device}, 4bit={use_4bit})...")

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )

            # 构建加载参数
            load_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
            }

            # CPU: 不传 device_map（避免内部触发 CUDA），float32
            if device == "cpu":
                load_kwargs["dtype"] = torch.float32
            else:
                load_kwargs["dtype"] = torch.float16
                load_kwargs["device_map"] = device  # "auto" | "cuda" | "cuda:0" 等

            if use_4bit and device != "cpu":
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                    logger.info("启用 4bit 量化 (bitsandbytes)")
                except ImportError:
                    logger.warning(
                        "bitsandbytes 未安装，跳过 4bit 量化。"
                        "安装方法: pip install bitsandbytes"
                    )

            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs,
            )

            self._model.eval()
            logger.info(f"本地模型加载完成: {model_name}")
            return True

        except Exception as e:
            self._load_failed = True
            self._load_failed_reason = (
                f"无法加载本地模型 '{model_name}'（{str(e)[:200]}）。"
                f"网络受限时请设置 HF_ENDPOINT=https://hf-mirror.com "
                f"或将 PARAM_EXTRACTION_MODE 设为 'local'（纯正则）或 'llm'。"
            )
            logger.warning(self._load_failed_reason)
            return False

    def _get_executor(self) -> ThreadPoolExecutor:
        """获取推理线程池"""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local_model")
        return self._executor

    def _build_extraction_prompt(
        self,
        extraction_prompt: str,
        message: str,
        schema_json: str,
    ) -> str:
        """
        构建参数抽取的 prompt，包含 JSON schema 定义。

        使用 Instruct 模型的 chat template 格式。
        """
        system = (
            "你是一个参数提取助手。从用户消息中提取结构化参数。\n"
            "严格按照以下 JSON Schema 输出 JSON，不要输出任何额外的文字或 markdown 代码块标记。\n"
            "如果用户没有提到某个字段，该字段值设为 null。\n"
            "只输出纯 JSON 对象，以 { 开头，以 } 结尾。\n\n"
            "【重要】JSON Schema 中每个字段后面的 // 注释是该字段的含义说明，不是需要填入的值！"
            "请只从「用户消息」中提取真实数据，不要把注释内容当作参数值填入。"
        )
        user = (
            f"提取规则:\n{self._clean_extraction_prompt(extraction_prompt)}\n\n"
            f"输出 JSON Schema（字段及类型说明）:\n{schema_json}\n\n"
            f"用户消息: {message}\n\n"
            "请按 JSON Schema 输出提取结果（纯 JSON, 不要 ``` 等标记）:"
        )

        # 使用 tokenizer 的 chat template（Qwen3-Instruct 格式）
        # enable_thinking=False: 参数抽取是简单提取任务，不需要 CoT 思维链
        if hasattr(self._tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except Exception:
                pass  # 降级到手动拼接

        # 手动拼接（兜底）
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

    # 剥离 Qwen3 等思考模型的 <think>...</think> 标签
    _STRIP_THINK_RE = re.compile(r'<think>.*?</think>\s*', re.DOTALL)

    def _strip_think_tags(self, text: str) -> str:
        """去除 Qwen3 等推理模型的 <think>...</think> 推理过程标签。"""
        return self._STRIP_THINK_RE.sub('', text).strip()

    def _extract_json_from_text(self, text: str) -> Optional[Dict]:
        """
        从模型输出文本中提取 JSON 对象。
        支持纯 JSON、带 markdown 代码块的 JSON、带 <think> 标签的 JSON。
        """
        # 0. 先剥离 <think>...</think> 标签（Qwen3 等推理模型会输出推理过程）
        text = self._strip_think_tags(text)

        # 1. 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. 提取 ```json ... ``` 代码块
        match = re.search(r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 3. 提取第一个 { ... } 对象（匹配最外层大括号）
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # 4. 更激进的提取：找最大括号范围
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        return None

    # 示例值剥离正则：去掉 description 中"，如 XXX"这类容易误导小模型的示例
    _STRIP_EXAMPLE_RE = re.compile(r'[，,]\s*如[：:]?\s*[^，,]+')
    # 提取规则中的示例剥离：去掉 "(如 202405270001)"、"(如 SF1234567890、YT123456)"
    _STRIP_PAREN_EXAMPLE_RE = re.compile(r'\(如[^)]+\)')

    @staticmethod
    def _clean_extraction_prompt(extraction_prompt: str) -> str:
        """剥离提取规则中的示例值，避免小模型照抄示例。"""
        return LocalModelService._STRIP_PAREN_EXAMPLE_RE.sub('', extraction_prompt)

    def _schema_to_json_desc(self, schema: Type[BaseModel]) -> str:
        """将 Pydantic Schema 转为简洁的 JSON Schema 描述。
        自动剥离 Field description 中的示例值（如 "订单号，如 WB202405270001" → "订单号"），
        避免小模型把示例当成实际参数填入。
        """
        model_schema = schema.model_json_schema()
        props = model_schema.get("properties", {})

        fields_desc = []
        for field_name, field_info in props.items():
            field_type = field_info.get("type", "string")
            field_desc = field_info.get("description", "")
            # 去掉示例值： "订单号，如 WB202405270001" → "订单号"
            if field_desc:
                field_desc = self._STRIP_EXAMPLE_RE.sub("", field_desc)
                # 去掉"筛选状态: 待付款/已发货/..."中的枚举值说明
                field_desc = re.sub(r'[：:]\s*[^，,]+$', '', field_desc)
            fields_desc.append(
                f'  "{field_name}": {field_type}  // {field_desc}'
            )

        return "{\n" + ",\n".join(fields_desc) + "\n}"

    @observe(name="local_model.extract_params")
    async def extract_params(
        self,
        extraction_prompt: str,
        message: str,
        output_schema: Type[BaseModel],
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        使用本地小模型提取结构化参数。

        Args:
            extraction_prompt: 提取规则说明
            message: 用户输入消息
            output_schema: Pydantic Schema（用于字段定义和验证）
            max_retries: JSON 解析失败时重试次数

        Returns:
            提取的参数 dict（已过滤 None 值）
        """
        t_total_start = _perf_time.perf_counter()

        t0 = _perf_time.perf_counter()
        ok = self._ensure_loaded()
        t_load = (_perf_time.perf_counter() - t0) * 1000
        if self._model is None:
            return {}  # 加载已失败，不重试，直接返回空

        # 构建 schema 描述
        t0 = _perf_time.perf_counter()
        schema_json = self._schema_to_json_desc(output_schema)
        t_schema = (_perf_time.perf_counter() - t0) * 1000

        for attempt in range(max_retries + 1):
            t_prompt_start = _perf_time.perf_counter()
            prompt = self._build_extraction_prompt(
                extraction_prompt,
                message,
                schema_json,
            )
            t_prompt = (_perf_time.perf_counter() - t_prompt_start) * 1000

            try:
                # 在独立线程中执行（transformers generate 是同步的）
                t_gen_start = _perf_time.perf_counter()
                raw_output = await asyncio.get_event_loop().run_in_executor(
                    self._get_executor(),
                    self._generate,
                    prompt,
                )
                t_gen_total = (_perf_time.perf_counter() - t_gen_start) * 1000

                # 提取 JSON
                t_json_start = _perf_time.perf_counter()
                data = self._extract_json_from_text(raw_output)
                t_json = (_perf_time.perf_counter() - t_json_start) * 1000

                if data is not None:
                    # 用 Pydantic 验证（容错：验证失败就手动过滤，不全丢弃）
                    t_valid_start = _perf_time.perf_counter()
                    try:
                        validated = output_schema(**data)
                        params = {
                            k: v for k, v in validated.model_dump().items()
                            if v is not None
                        }
                    except Exception as ve:
                        logger.warning(
                            f"Schema 验证失败（尝试 {attempt + 1}）: {str(ve)[:100]}, "
                            f"回退为手动过滤"
                        )
                        # 手动过滤：只保留 schema 中定义的字段
                        valid_fields = set(output_schema.model_fields.keys())
                        params = {
                            k: v for k, v in data.items()
                            if k in valid_fields and v is not None
                        }
                    t_valid = (_perf_time.perf_counter() - t_valid_start) * 1000

                    if params:
                        t_total = (_perf_time.perf_counter() - t_total_start) * 1000
                        logger.debug(
                            "本地模型参数提取成功 [耗时统计]",
                            total_ms=round(t_total, 1),
                            load_ms=round(t_load, 1),
                            schema_ms=round(t_schema, 1),
                            prompt_ms=round(t_prompt, 1),
                            generate_ms=round(t_gen_total, 1),
                            json_parse_ms=round(t_json, 1),
                            validate_ms=round(t_valid, 1),
                            attempt=attempt + 1,
                            extracted=list(params.keys()),
                        )
                        logger.info(
                            "本地模型参数提取成功",
                            extracted=list(params.keys()),
                            attempt=attempt + 1,
                            duration_total_ms=round(t_total, 1),
                            duration_load_ms=round(t_load, 1),
                            duration_schema_ms=round(t_schema, 1),
                            duration_prompt_ms=round(t_prompt, 1),
                            duration_generate_ms=round(t_gen_total, 1),
                            duration_json_parse_ms=round(t_json, 1),
                            duration_validate_ms=round(t_valid, 1),
                        )
                        return params

                # JSON 解析失败或空结果
                if attempt < max_retries:
                    logger.warning(
                        f"JSON 解析失败（尝试 {attempt + 1}）: "
                        f"raw={raw_output[:150]}, "
                        f"prompt_ms={round(t_prompt, 1)}, "
                        f"generate_ms={round(t_gen_total, 1)}, "
                        f"json_parse_ms={round(t_json, 1)}"
                    )
                else:
                    logger.warning(
                        f"本地模型参数提取失败（已重试 {max_retries} 次），返回空参数"
                    )

            except Exception as e:
                logger.error(
                    f"本地模型推理异常（尝试 {attempt + 1}）: {str(e)[:200]}"
                )

        return {}

    def _generate(self, prompt: str) -> str:
        """同步生成（在 executor 线程中执行），含粒度耗时统计"""
        import torch

        g_start = _perf_time.perf_counter()

        # 1. Tokenization
        tok_start = _perf_time.perf_counter()
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        t_tok = (_perf_time.perf_counter() - tok_start) * 1000
        input_len = inputs["input_ids"].shape[1]

        # 2. 移到设备
        dev_start = _perf_time.perf_counter()
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        t_dev = (_perf_time.perf_counter() - dev_start) * 1000

        # 3. 推理
        inf_start = _perf_time.perf_counter()
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=chat_config.local_param_max_tokens,
                do_sample=False,  # 参数抽取用贪心解码，输出稳定
                temperature=None,  # do_sample=False 时忽略
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        t_infer = (_perf_time.perf_counter() - inf_start) * 1000

        # 4. 解码
        dec_start = _perf_time.perf_counter()
        # 只取新生成的部分（不包含输入 prompt）
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        output_len = generated_ids.shape[0]
        result = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        t_dec = (_perf_time.perf_counter() - dec_start) * 1000

        g_total = (_perf_time.perf_counter() - g_start) * 1000

        logger.debug(
            "本地模型 _generate",
            input_tokens=input_len,
            output_tokens=output_len,
            duration_total_ms=round(g_total, 1),
            duration_tokenize_ms=round(t_tok, 1),
            duration_to_device_ms=round(t_dev, 1),
            duration_infer_ms=round(t_infer, 1),
            duration_decode_ms=round(t_dec, 1),
        )

        return result.strip()

    # ════════════════════════════════════════════════════════════════
    # 工具选择器专用模型（独立于参数抽取模型）
    # ════════════════════════════════════════════════════════════════

    def _ensure_tool_selector_loaded(self) -> bool:
        """加载工具选择器专用本地模型。返回 True 表示可用。"""
        if self._tool_selector_model is not None:
            return True
        if self._tool_selector_load_failed:
            return False

        model_path = (chat_config.tool_selector_local_model or "").strip()
        if not model_path:
            self._tool_selector_load_failed = True
            self._tool_selector_load_failed_reason = "TOOL_SELECTOR_LOCAL_MODEL 未配置"
            return False

        device = chat_config.tool_selector_local_device
        use_4bit = chat_config.tool_selector_local_load_in_4bit

        logger.info(f"正在加载工具选择器本地模型: {model_path} (device={device}, 4bit={use_4bit})...")

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            self._tool_selector_tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True,
            )
            load_kwargs: Dict[str, Any] = {"trust_remote_code": True}

            if device == "cpu":
                load_kwargs["dtype"] = torch.float32
            else:
                load_kwargs["dtype"] = torch.float16
                load_kwargs["device_map"] = device

            if use_4bit and device != "cpu":
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                except ImportError:
                    logger.warning("bitsandbytes 未安装，跳过 4bit 量化")

            self._tool_selector_model = AutoModelForCausalLM.from_pretrained(
                model_path, **load_kwargs,
            )
            self._tool_selector_model.eval()
            logger.info(f"工具选择器模型加载完成: {model_path}")
            return True

        except Exception as e:
            self._tool_selector_load_failed = True
            self._tool_selector_load_failed_reason = str(e)[:300]
            logger.warning(
                f"工具选择器模型加载失败: {model_path} — {str(e)[:200]}。"
                f"将回退到云端 P2 模型。"
            )
            return False

    @observe(name="local_model.chat_classify")
    async def chat_classify(
        self,
        user_query: str,
        tool_names: list[str],
        tool_descriptions: dict[str, str],
        system_prompt: str = "",
        max_retries: int = 1,
    ) -> list[str]:
        """使用工具选择器本地模型从候选工具列表中选出最相关的。

        构建 prompt → 本地推理 → 解析输出中的工具名列表。

        Args:
            user_query: 用户原始消息
            tool_names: 候选工具名列表（P0+P1 过滤后的结果，通常 3-5 个）
            tool_descriptions: {工具名: 描述} 映射
            system_prompt: 路由规则说明
            max_retries: 解析失败重试次数

        Returns:
            选中的工具名列表（保持原始排序）
        """
        t_start = _perf_time.monotonic()

        if not self._ensure_tool_selector_loaded():
            # 本地模型不可用 —— 返回全部候选（让上层回退到 P1 结果）
            logger.debug("工具选择器本地模型不可用，返回全部候选工具")
            return list(tool_names)

        if len(tool_names) <= 2:
            return list(tool_names)  # 已经足够少，无需再选

        # 构建工具列表描述
        tool_lines = "\n".join(
            f"- {name}: {tool_descriptions.get(name, '')}"
            for name in tool_names
        )

        # 构建 prompt（使用 chat template）
        if not system_prompt:
            system_prompt = (
                "你是一个工具路由器。根据用户消息从候选工具里选出最相关的工具。"
                "只输出工具名，每行一个，不要任何解释或标记。"
            )
        user_msg = (
            f"候选工具:\n{tool_lines}\n\n"
            f"用户消息: {user_query}\n\n"
            f"请选出最相关的工具（只输出工具名，每行一个）:"
        )

        # 使用 tokenizer 的 chat template
        if hasattr(self._tool_selector_tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]
            try:
                prompt = self._tool_selector_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,  # 工具选择是简单分类任务，不需要 CoT
                )
            except Exception:
                prompt = (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
        else:
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        # 合法的工具名集合
        valid_names = set(tool_names)

        for attempt in range(max_retries + 1):
            try:
                t_gen_start = _perf_time.monotonic()
                raw_output = await asyncio.get_event_loop().run_in_executor(
                    self._get_executor(),
                    self._tool_selector_generate,
                    prompt,
                )
                t_gen = (_perf_time.monotonic() - t_gen_start) * 1000

                # 剥离 <think>...</think> 标签（Qwen3 等推理模型输出）
                raw_output = self._strip_think_tags(raw_output)

                # 解析: 每行提取工具名（忽略空行、标点、多余空格）
                selected = []
                for line in raw_output.strip().splitlines() if raw_output else []:
                    name = line.strip().lstrip("-* 0123456789.、，").strip()
                    # 去掉可能的引号、逗号等
                    name = name.strip('\'"`,，:')
                    if name and name in valid_names and name not in selected:
                        selected.append(name)

                if selected:
                    t_total = (_perf_time.monotonic() - t_start) * 1000
                    logger.info(
                        "工具选择器本地模型完成",
                        candidate_count=len(tool_names),
                        selected_count=len(selected),
                        selected=selected,
                        generate_ms=round(t_gen, 1),
                        total_ms=round(t_total, 1),
                        attempt=attempt + 1,
                    )
                    return selected

                # 没解析到合法工具名，降级为保留全部候选
                logger.warning(
                    f"工具选择器解析失败 (attempt {attempt + 1}), "
                    f"raw={raw_output[:120]}"
                )

            except Exception as e:
                logger.error(f"工具选择器推理异常 (attempt {attempt + 1}): {str(e)[:200]}")

        # 降至全部候选（回退到 P1 结果）
        return list(tool_names)

    def _tool_selector_generate(self, prompt: str) -> str:
        """同步生成（在 executor 线程中执行，工具选择器模型专用）。"""
        import torch

        inputs = self._tool_selector_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048,
        )
        device = next(self._tool_selector_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._tool_selector_model.generate(
                **inputs,
                max_new_tokens=64,  # 工具选择输出很短
                do_sample=False,
                pad_token_id=self._tool_selector_tokenizer.eos_token_id,
                eos_token_id=self._tool_selector_tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tool_selector_tokenizer.decode(
            generated_ids, skip_special_tokens=True,
        ).strip()

    # ════════════════════════════════════════════════════════════════
    # Step2 安全审查分类（复用参数抽取模型，二进制分类）
    # ════════════════════════════════════════════════════════════════

    @observe(name="local_model.safety_classify")
    async def safety_classify(
        self,
        system_prompt: str,
        user_question: str,
        max_retries: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """使用本地小模型做安全合规分类（Step2 内容审查加速）。

        Args:
            system_prompt: 合规判断规则说明（与云端 LLM 共用同一 prompt）
            user_question: 用户问题文本
            max_retries: JSON 解析失败重试次数

        Returns:
            解析后的合规判断 dict（如 {"compliant": true, "issue": ""}），
            None 表示模型不可用或解析失败（调用方应回退到云端 LLM）
        """
        t_start = _perf_time.monotonic()

        ok = self._ensure_loaded()
        if not ok or self._model is None:
            logger.debug("安全审查本地模型不可用，跳过")
            return None

        # 构建 prompt（短文本二分类，不需要思维链）
        user_msg = (
            f"{system_prompt}\n\n"
            f"用户问题：{user_question}\n\n"
            f"请严格按JSON格式输出判断结果，不要输出任何额外文字："
        )

        if hasattr(self._tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": "你是一个内容安全审查助手。只输出JSON，不要解释。"},
                {"role": "user", "content": user_msg},
            ]
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,  # 简单分类，不需要 CoT
                )
            except Exception:
                prompt = (
                    f"<|im_start|>system\n你是一个内容安全审查助手。只输出JSON，不要解释。<|im_end|>\n"
                    f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
        else:
            prompt = (
                f"<|im_start|>system\n你是一个内容安全审查助手。只输出JSON，不要解释。<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        for attempt in range(max_retries + 1):
            try:
                t_gen_start = _perf_time.monotonic()
                raw_output = await asyncio.get_event_loop().run_in_executor(
                    self._get_executor(),
                    self._safety_generate,
                    prompt,
                )
                t_gen = (_perf_time.monotonic() - t_gen_start) * 1000

                # 剥离 <think>...</think>
                raw_output = self._strip_think_tags(raw_output)

                # JSON 解析
                data = self._extract_json_from_text(raw_output)
                if data is not None:
                    t_total = (_perf_time.monotonic() - t_start) * 1000
                    logger.info(
                        "本地模型安全审查完成",
                        duration_total_ms=round(t_total, 1),
                        duration_generate_ms=round(t_gen, 1),
                        attempt=attempt + 1,
                        result=data,
                    )
                    return data

                # JSON 解析失败
                if attempt < max_retries:
                    logger.warning(
                        f"安全审查 JSON 解析失败 (attempt {attempt + 1}): "
                        f"raw={raw_output[:120]}"
                    )
                else:
                    logger.warning(
                        f"安全审查本地模型失败（已重试 {max_retries} 次），回退到云端LLM"
                    )

            except Exception as e:
                logger.error(f"安全审查本地模型异常 (attempt {attempt + 1}): {str(e)[:200]}")

        return None  # 回退到云端 LLM

    def _safety_generate(self, prompt: str) -> str:
        """同步生成（executor 线程中执行，安全审查专用）。"""
        import torch

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=64,  # 安全审查输出很短：{"compliant":true,"issue":""}
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(
            generated_ids, skip_special_tokens=True,
        ).strip()

    def close(self):
        """释放模型资源"""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        if self._model is not None:
            del self._model
            self._model = None
            import gc
            gc.collect()
            if hasattr(self, "_tokenizer") and self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
            logger.info("本地模型已释放")

        if self._tool_selector_model is not None:
            del self._tool_selector_model
            self._tool_selector_model = None
            if hasattr(self, "_tool_selector_tokenizer") and self._tool_selector_tokenizer is not None:
                del self._tool_selector_tokenizer
                self._tool_selector_tokenizer = None
            logger.info("工具选择器模型已释放")

        self._load_failed = False
        self._load_failed_reason = ""
        self._tool_selector_load_failed = False
        self._tool_selector_load_failed_reason = ""
        LocalModelService._instance = None
