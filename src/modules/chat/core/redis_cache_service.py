"""
Redis 向量缓存服务
使用 Redis Stack 的向量相似度搜索（VSS）功能
存储最近对话历史和高频问题，实现问题去重
"""

import hashlib
import json
import re
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import redis
import numpy as np

from src.shared.logger import APILogger
from src.modules.chat.config import chat_config, ChatConfig

logger = APILogger("redis_cache_service")

# Redis Key 前缀
KEY_PREFIX = "hospital_chat:"
QUESTION_EMBEDDINGS_KEY = f"{KEY_PREFIX}question_embeddings"  # 向量索引
CONVERSATION_HISTORY_KEY = f"{KEY_PREFIX}conversation:{{conversation_id}}"  # 对话历史
FREQUENT_QUESTIONS_KEY = f"{KEY_PREFIX}frequent_questions"  # 高频问题
QUESTION_CACHE_KEY = f"{KEY_PREFIX}question_cache"  # 问题缓存（用于快速查找）

# 配置（从 chat_config 读取，无配置时使用默认值）
MAX_RECENT_CONVERSATIONS = 5  # 最近对话数量


class RedisCacheService:
    """Redis 向量缓存服务（单例）"""
    
    _instance: Optional["RedisCacheService"] = None
    _initialized: bool = False
    
    def __new__(cls) -> "RedisCacheService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._client = None  # 显式在实例上初始化
            cls._instance._initialized = False
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> "RedisCacheService":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self) -> None:
        if not self._initialized:
            self._initialize()
            self._initialized = True
    
    def _get_config(self) -> "ChatConfig":
        """获取配置"""
        return chat_config
    
    def _sanitize_question(self, question: str) -> str:
        """
        对用户问题进行脱敏处理
        移除敏感个人信息
        """
        # 移除电话号码（7位以上连续数字）
        question = re.sub(r'\d{7,}', '[电话]', question)
        # 移除邮箱
        question = re.sub(r'\w+@\w+\.\w+', '[邮箱]', question)
        # 移除身份证号
        question = re.sub(r'\d{15,17}[\dXx]', '[证件号]', question)
        # 移除详细地址
        question = re.sub(r'[^,，。；;]{10,}?(?:街|路|巷|号|楼|层|室|栋|单元)', '', question)
        return question.strip()
    
    def _get_question_hash(self, question: str) -> str:
        """使用 SHA256 生成问题哈希（避免哈希冲突）"""
        normalized = question.strip().lower()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]
    
    def _initialize(self) -> None:
        """初始化 Redis 连接"""
        try:
            config = self._get_config()
            
            # 从配置获取 Redis 连接参数
            redis_host = config.redis_host or "localhost"
            redis_port = config.redis_port or 6379
            redis_password = getattr(config, "redis_password", "") or None
            
            self._client = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                db=0,
                decode_responses=False,  # 向量数据需要二进制
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # 测试连接
            self._client.ping()
            
            # 初始化向量索引
            self._ensure_index_exists()
            
            logger.info(f"Redis 连接成功: {redis_host}:{redis_port}")
            
        except redis.ConnectionError as e:
            logger.warning(f"Redis 连接失败，将禁用缓存功能: {str(e)}")
            self._client = None
        except redis.TimeoutError as e:
            logger.warning(f"Redis 连接超时: {str(e)}")
            self._client = None
        except Exception as e:
            logger.error(f"Redis 初始化失败: {str(e)}")
            self._client = None
    
    def _ensure_index_exists(self) -> None:
        """确保 Redis 向量索引存在"""
        if self._client is None:
            return
        
        try:
            config = self._get_config()
            embedding_dim = config.embedding_dimension or 2048
            
            # 先检查索引是否已存在（使用 FT.INFO 更可靠）
            try:
                self._client.execute_command("FT.INFO", "hospital_questions_idx")
                logger.debug("Redis 向量索引已存在: hospital_questions_idx")
                return
            except redis.ResponseError:
                pass  # 索引不存在，继续创建
            
            # 创建向量索引（使用 COSINE 相似度）
            # Redis Stack 7.x VECTOR 语法: VECTOR [algorithm] [number_of_params] [params...]
            self._client.execute_command(
                "FT.CREATE", "hospital_questions_idx",
                "ON", "HASH",
                "PREFIX", "1", f"{KEY_PREFIX}q:",
                "SCHEMA",
                "question", "TEXT",
                "answer", "TEXT",
                "embedding", "VECTOR", "FLAT", "6", 
                "TYPE", "FLOAT64", "DIM", str(embedding_dim), "DISTANCE_METRIC", "COSINE",
                "timestamp", "NUMERIC",
                "conversation_id", "TEXT"
            )
            logger.info(f"创建 Redis 向量索引: hospital_questions_idx (dim={embedding_dim})")
            
        except redis.ResponseError as e:
            if "Index already exists" in str(e):
                logger.debug("Redis 向量索引已存在（并发创建）: hospital_questions_idx")
            else:
                logger.warning(f"检查/创建索引失败（非致命）: {str(e)}")
        except redis.RedisError as e:
            logger.warning(f"Redis 连接错误: {str(e)}")
    
    @property
    def is_available(self) -> bool:
        """检查 Redis 是否可用"""
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except (redis.ConnectionError, redis.TimeoutError):
            return False
        except Exception:
            return False
    
    def _vector_to_bytes(self, vector: List[float]) -> bytes:
        """将向量转换为字节"""
        return np.array(vector, dtype=np.float64).tobytes()
    
    def _bytes_to_vector(self, data: bytes) -> List[float]:
        """将字节转换为向量"""
        return np.frombuffer(data, dtype=np.float64).tolist()
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    async def store_conversation(
        self,
        conversation_id: str,
        question: str,
        answer: str,
        embedding: List[float]
    ) -> bool:
        """
        存储对话到 Redis
        
        Args:
            conversation_id: 会话ID
            question: 用户问题
            answer: 回答
            embedding: 问题向量
            
        Returns:
            是否存储成功
        """
        if not self.is_available:
            return False
        
        try:
            config = self._get_config()
            cache_expire_seconds = (config.cache_expire_days or 7) * 24 * 3600
            timestamp = int(time.time())
            cache_id = f"{KEY_PREFIX}q:{conversation_id}_{timestamp}"
            
            # 存储为 Hash
            pipe = self._client.pipeline()
            
            # 存储问题-回答对（用于向量搜索）
            pipe.hset(cache_id, mapping={
                "question": question.encode('utf-8'),
                "answer": answer.encode('utf-8'),
                "embedding": self._vector_to_bytes(embedding),
                "timestamp": str(timestamp).encode('utf-8'),
                "conversation_id": conversation_id.encode('utf-8')
            })
            pipe.expire(cache_id, cache_expire_seconds)
            
            # 更新对话历史（List）
            history_key = CONVERSATION_HISTORY_KEY.format(conversation_id=conversation_id)
            conversation_data = json.dumps({
                "question": question,
                "answer": answer,
                "timestamp": timestamp
            }, ensure_ascii=False)
            pipe.rpush(history_key, conversation_data)
            # 只保留最近5条
            pipe.ltrim(history_key, -MAX_RECENT_CONVERSATIONS * 2, -1)
            pipe.expire(history_key, cache_expire_seconds)
            
            # 更新高频问题计数（Sorted Set）
            question_normalized = question.strip().lower()
            pipe.zincrby(FREQUENT_QUESTIONS_KEY, 1, question_normalized)
            
            # 存储原始问题到缓存（用于快速查找）- 使用 SHA256 哈希
            cache_key = f"{QUESTION_CACHE_KEY}:{self._get_question_hash(question)}"
            pipe.hset(cache_key, mapping={
                "question": question,
                "answer": answer,
                "embedding": self._vector_to_bytes(embedding)
            })
            pipe.expire(cache_key, cache_expire_seconds)
            
            pipe.execute()
            
            logger.info(f"对话已存储: conversation_id={conversation_id}, question_len={len(question)}")
            return True
            
        except redis.RedisError as e:
            logger.error(f"存储对话失败: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"存储对话失败: {str(e)}")
            return False
    
    async def search_similar_questions(
        self,
        question_embedding: List[float],
        threshold: Optional[float] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        搜索相似问题（使用 FT.SEARCH 向量索引）
        
        Args:
            question_embedding: 问题向量
            threshold: 相似度阈值（默认从配置读取）
            top_k: 返回数量
            
        Returns:
            相似问题列表 [{question, answer, similarity}]
        """
        if not self.is_available:
            logger.warning("Redis 不可用，跳过向量相似度搜索")
            return []
        
        try:
            config = self._get_config()
            threshold = threshold or config.redis_vector_threshold or 0.85
            vector_bytes = self._vector_to_bytes(question_embedding)
            
            # 尝试使用 FT.SEARCH 进行高效向量搜索
            try:
                result = self._client.execute_command(
                    "FT.SEARCH", "hospital_questions_idx",
                    f"*=>[KNN {top_k * 2} @embedding $vec AS similarity]",
                    "PARAMS", "2", "vec", vector_bytes,
                    "RETURN", "3", "question", "answer", "similarity",
                    "DIALECT", "2",
                    "LIMIT", "0", str(top_k * 2)
                )
                
                similar_questions = []
                if result and len(result) > 1:
                    # result[0] 是总数量，result[1:] 是结果
                    for item in result[1:]:
                        try:
                            # item 格式: [key, field1, value1, field2, value2, ...]
                            # 解析字段-值对
                            fields = {}
                            for i in range(1, len(item), 2):
                                if i + 1 >= len(item):
                                    break
                                field = item[i].decode('utf-8') if isinstance(item[i], bytes) else str(item[i])
                                value = item[i + 1]
                                
                                if field == "question":
                                    fields["question"] = value.decode('utf-8') if isinstance(value, bytes) else str(value)
                                elif field == "answer":
                                    fields["answer"] = value.decode('utf-8') if isinstance(value, bytes) else str(value)
                                elif field == "similarity":
                                    # similarity 可能是字节字符串或其他格式
                                    if isinstance(value, bytes):
                                        value = value.decode('utf-8')
                                    fields["similarity"] = float(value)
                            
                            if "question" in fields and "answer" in fields:
                                # Redis 返回的 similarity 是距离，取反得到相似度
                                similarity = fields.get("similarity", 0.0)
                                if isinstance(similarity, (int, float)):
                                    similarity = max(0.0, 1.0 - float(similarity))
                                else:
                                    similarity = 0.0
                                
                                if similarity >= threshold:
                                    similar_questions.append({
                                        "question": fields["question"],
                                        "answer": fields["answer"],
                                        "similarity": round(similarity, 4),
                                        "conversation_id": fields.get("conversation_id", "")
                                    })
                        except (ValueError, IndexError, TypeError) as parse_error:
                            logger.debug(f"解析搜索结果项失败: {parse_error}")
                            continue
                
                similar_questions.sort(key=lambda x: x["similarity"], reverse=True)
                logger.info(f"使用 FT.SEARCH 找到 {len(similar_questions)} 个相似问题（阈值={threshold}）")
                return similar_questions[:top_k]
                
            except redis.RedisError:
                # FT.SEARCH 不可用，回退到 SCAN + 余弦相似度
                logger.warning("FT.SEARCH 不可用，回退到 SCAN 方式")
                return await self._search_similar_questions_fallback(question_embedding, threshold, top_k)
            
        except Exception as e:
            logger.error(f"搜索相似问题失败: {str(e)}")
            return []
    
    async def _search_similar_questions_fallback(
        self,
        question_embedding: List[float],
        threshold: float = 0.85,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """回退方案：使用 SCAN 遍历所有问题向量进行相似度计算"""
        similar_questions = []
        cursor = 0
        
        while True:
            cursor, keys = self._client.scan(cursor, match=f"{KEY_PREFIX}q:*", count=100)
            
            for key in keys:
                try:
                    data = self._client.hgetall(key)
                    if not data or b"embedding" not in data:
                        continue
                    
                    stored_embedding = self._bytes_to_vector(data[b"embedding"])
                    similarity = self._cosine_similarity(question_embedding, stored_embedding)
                    
                    if similarity >= threshold:
                        similar_questions.append({
                            "question": data[b"question"].decode('utf-8'),
                            "answer": data[b"answer"].decode('utf-8'),
                            "similarity": round(similarity, 4),
                            "conversation_id": data.get(b"conversation_id", b"").decode('utf-8')
                        })
                except Exception:
                    continue
            
            if cursor == 0:
                break
        
        # 按相似度排序
        similar_questions.sort(key=lambda x: x["similarity"], reverse=True)
        logger.info(f"使用 SCAN 回退找到 {len(similar_questions)} 个相似问题（阈值={threshold}）")
        return similar_questions[:top_k]
    
    async def get_cached_response(
        self,
        question: str,
        question_embedding: List[float],
        threshold: Optional[float] = None
    ) -> Optional[str]:
        """
        获取缓存的回答（如果存在相似问题）
        
        Args:
            question: 用户问题
            question_embedding: 问题向量
            threshold: 相似度阈值（默认从配置读取）
            
        Returns:
            缓存的回答，如果没有相似问题则返回 None
        """
        # 先尝试精确匹配
        cache_key = f"{QUESTION_CACHE_KEY}:{self._get_question_hash(question)}"
        
        try:
            if self.is_available:
                cached = self._client.hgetall(cache_key)
                if cached and b"answer" in cached:
                    logger.info(f"精确命中缓存: {question[:30]}...")
                    return cached[b"answer"].decode('utf-8')
        except redis.RedisError as e:
            logger.warning(f"精确匹配失败: {str(e)}")
        
        # 再尝试向量相似度搜索
        similar = await self.search_similar_questions(question_embedding, threshold, top_k=1)
        if similar:
            logger.info(f"向量相似度命中: similarity={similar[0]['similarity']}, question={similar[0]['question'][:30]}...")
            return similar[0]["answer"]
        
        return None
    
    def get_recent_conversations(
        self,
        conversation_id: str,
        limit: int = MAX_RECENT_CONVERSATIONS
    ) -> List[Dict[str, Any]]:
        """
        获取最近的对话历史
        
        Args:
            conversation_id: 会话ID
            limit: 返回数量
            
        Returns:
            对话历史列表
        """
        if not self.is_available:
            return []
        
        try:
            history_key = CONVERSATION_HISTORY_KEY.format(conversation_id=conversation_id)
            items = self._client.lrange(history_key, -limit * 2, -1)
            
            conversations = []
            for item in items:
                try:
                    data = json.loads(item)
                    conversations.append(data)
                except json.JSONDecodeError:
                    continue
            
            return conversations
            
        except redis.RedisError as e:
            logger.error(f"获取对话历史失败: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"获取对话历史失败: {str(e)}")
            return []
    
    # =========================================================================
    # Agent 对话历史（role+content 格式，供 GeneralAgentExecutor 使用）
    # =========================================================================
    
    CHAT_HISTORY_KEY = f"{KEY_PREFIX}chat_history:{{conversation_id}}"
    
    def add_chat_message(self, conversation_id: str, role: str, content: str,
                         max_turns: int = 10, expire_days: int = 1) -> bool:
        """向 Redis 追加一条 Agent 对话消息，并裁剪到 max_turns 轮。
        
        Args:
            conversation_id: 会话ID
            role: 消息角色（user/assistant/system）
            content: 消息内容
            max_turns: 最大保留轮数（每轮 user+assistant 占 2 条），超出的自动裁剪
            expire_days: 过期天数，默认 1 天
            
        Returns:
            是否存储成功
        """
        if not self.is_available:
            return False
        
        try:
            history_key = self.CHAT_HISTORY_KEY.format(conversation_id=conversation_id)
            message = json.dumps({"role": role, "content": content}, ensure_ascii=False)
            pipe = self._client.pipeline()
            pipe.rpush(history_key, message)
            # 保留最近 max_turns * 2 条（每轮一对 user+assistant）
            pipe.ltrim(history_key, -max_turns * 2, -1)
            pipe.expire(history_key, expire_days * 86400)
            pipe.execute()
            return True
        except redis.RedisError as e:
            logger.error(f"追加对话消息失败: {str(e)}")
            return False
    
    def get_chat_messages(self, conversation_id: str, max_turns: int = 10
                          ) -> List[Dict[str, str]]:
        """从 Redis 获取 Agent 对话历史（role+content 格式）。
        
        Args:
            conversation_id: 会话ID
            max_turns: 最大返回轮数
            
        Returns:
            消息列表 [{"role": "user", "content": "..."}, ...]
        """
        if not self.is_available:
            return []
        
        try:
            history_key = self.CHAT_HISTORY_KEY.format(conversation_id=conversation_id)
            items = self._client.lrange(history_key, -max_turns * 2, -1)
            
            messages = []
            for item in items:
                try:
                    msg = json.loads(item)
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        messages.append(msg)
                except json.JSONDecodeError:
                    continue
            
            return messages
        except redis.RedisError as e:
            logger.error(f"获取对话消息失败: {str(e)}")
            return []
    
    def get_frequent_questions(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        获取高频问题
        
        Args:
            top_n: 返回数量
            
        Returns:
            高频问题列表 [{question, count}]
        """
        if not self.is_available:
            return []
        
        try:
            # 获取高频问题（从 Sorted Set）
            items = self._client.zrevrange(FREQUENT_QUESTIONS_KEY, 0, top_n - 1, withscores=True)
            
            frequent_questions = []
            for question, count in items:
                question_str = question.decode('utf-8') if isinstance(question, bytes) else question
                # 获取详细信息
                cache_key = f"{QUESTION_CACHE_KEY}:{self._get_question_hash(question_str)}"
                cached = self._client.hgetall(cache_key)
                
                if cached and b"answer" in cached:
                    frequent_questions.append({
                        "question": cached[b"question"].decode('utf-8'),
                        "answer": cached[b"answer"].decode('utf-8'),
                        "count": int(count)
                    })
                else:
                    frequent_questions.append({
                        "question": question_str,
                        "count": int(count)
                    })
            
            return frequent_questions
            
        except redis.RedisError as e:
            logger.error(f"获取高频问题失败: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"获取高频问题失败: {str(e)}")
            return []
    
    async def cleanup_old_conversations(self, max_age_days: int = 30) -> int:
        """
        清理过期的对话
        
        Args:
            max_age_days: 最大保留天数
            
        Returns:
            清理的对话数量
        """
        if not self.is_available:
            return 0
        
        try:
            cutoff_time = int((datetime.now() - timedelta(days=max_age_days)).timestamp())
            deleted_count = 0
            
            cursor = 0
            while True:
                cursor, keys = self._client.scan(cursor, match=f"{KEY_PREFIX}q:*", count=100)
                
                for key in keys:
                    try:
                        timestamp = self._client.hget(key, "timestamp")
                        if timestamp:
                            ts = int(timestamp.decode('utf-8'))
                            if ts < cutoff_time:
                                self._client.delete(key)
                                deleted_count += 1
                    except Exception:
                        continue
                
                if cursor == 0:
                    break
            
            logger.info(f"清理了 {deleted_count} 条过期对话")
            return deleted_count
            
        except redis.RedisError as e:
            logger.error(f"清理过期对话失败: {str(e)}")
            return 0
        except Exception as e:
            logger.error(f"清理过期对话失败: {str(e)}")
            return 0
    
    def close(self) -> None:
        """关闭 Redis 连接"""
        if self._client:
            try:
                self._client.close()
                self._client = None
                logger.info("Redis 连接已关闭")
            except Exception as e:
                logger.error(f"关闭 Redis 连接失败: {str(e)}")


# 全局单例
_redis_cache_service: Optional[RedisCacheService] = None


def get_redis_cache_service() -> RedisCacheService:
    """获取 Redis 缓存服务单例"""
    global _redis_cache_service
    if _redis_cache_service is None:
        _redis_cache_service = RedisCacheService.get_instance()
    return _redis_cache_service
