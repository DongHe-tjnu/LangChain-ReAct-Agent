"""长期记忆的应用服务与故障隔离边界。"""

from __future__ import annotations

import hashlib
import time

from memory.contracts import MemoryProvider
from memory.models import MemoryRecord, RecallResult
from utils.logger_handler import logger


REFLECTION_QUERY_TEMPLATE = """请基于该 Memory Bank 中已经积累的长期记忆，生成与当前问题相关的用户背景摘要。
优先总结：用户画像、稳定偏好、长期兴趣、持续项目、重要经历、长期任务与目标。
只使用已有记忆；没有相关信息时返回空内容或明确表示没有相关记忆。

当前用户问题：{query}"""


class MemoryManager:
    """组织多用户 Bank 路由，并保证 Memory 故障不影响 Agent。"""

    def __init__(
        self,
        provider: MemoryProvider,
        bank_id_prefix: str,
        reflection_enabled: bool = True,
        search_limit: int = 20,
        enabled: bool = True,
    ) -> None:
        self._provider = provider
        self._bank_id_prefix = bank_id_prefix.strip() or "langchain-react-agent"
        self._reflection_enabled = reflection_enabled
        self._search_limit = search_limit
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Memory 是否已启用且 Provider 初始化成功。"""
        return self._enabled

    def bank_id_for(self, user_id: str) -> str:
        """将稳定用户 ID 映射为不泄露原始标识的独立 Bank ID。"""
        normalized_user_id = user_id.strip()
        if not normalized_user_id:
            raise ValueError("user_id不能为空")
        digest = hashlib.sha256(normalized_user_id.encode("utf-8")).hexdigest()[:24]
        return f"{self._bank_id_prefix}:user:{digest}"

    def recall(self, user_id: str, query: str, include_reflection: bool = False) -> RecallResult:
        """召回用户记忆；任意异常均降级为空结果。"""
        started_at = time.perf_counter()
        try:
            bank_id = self.bank_id_for(user_id)
            memories = self._provider.recall(bank_id, query)
            logger.info(
                "[Memory Recall] bank_id=%s count=%s elapsed_ms=%.1f",
                bank_id,
                len(memories),
                (time.perf_counter() - started_at) * 1000,
            )
        except Exception as exc:
            logger.warning(
                "[Memory Recall] 调用失败，Agent将无记忆继续回答：%s: %r",
                type(exc).__name__,
                exc,
            )
            memories = ()

        reflection = None
        # 空 Bank 没有可总结内容，跳过一次昂贵且无意义的 Reflection。
        if include_reflection and self._reflection_enabled and memories:
            reflection_query = REFLECTION_QUERY_TEMPLATE.format(query=query.strip())
            reflection = self.reflect(user_id, reflection_query)

        return RecallResult(memories=memories, reflection=reflection)

    def retain(self, user_id: str, memory: MemoryRecord) -> bool:
        """保存一条已提炼记忆；失败时返回 False。"""
        try:
            bank_id = self.bank_id_for(user_id)
            self._provider.retain(bank_id, memory)
            logger.info("[Memory Retain] bank_id=%s type=%s success=true", bank_id, memory.memory_type.value)
            return True
        except Exception as exc:
            logger.warning(
                "[Memory Retain] 调用失败，已跳过：%s: %r",
                type(exc).__name__,
                exc,
            )
            return False

    def retain_many(self, user_id: str, memories: tuple[MemoryRecord, ...]) -> int:
        """逐条保存提炼后的记忆，并返回成功数量。"""
        if not self._enabled:
            return 0
        return sum(self.retain(user_id, memory) for memory in memories)

    def reflect(self, user_id: str, query: str) -> str | None:
        """生成用户长期画像；失败时返回 None。"""
        try:
            bank_id = self.bank_id_for(user_id)
            result = self._provider.reflect(bank_id, query)
            logger.info("[Memory Reflect] bank_id=%s success=%s", bank_id, bool(result))
            return result
        except Exception as exc:
            logger.warning(
                "[Memory Reflect] 调用失败，已跳过：%s: %r",
                type(exc).__name__,
                exc,
            )
            return None

    def search(self, user_id: str, query: str) -> tuple[MemoryRecord, ...]:
        """搜索用户 Bank；失败时返回空集合。"""
        try:
            return self._provider.search(self.bank_id_for(user_id), query, self._search_limit)
        except Exception as exc:
            logger.warning("[Memory Search] 调用失败，已返回空结果：%s", exc)
            return ()

    def delete(self, user_id: str, document_id: str) -> bool:
        """安全删除指定文档，绝不隐式清空整个 Bank。"""
        try:
            return self._provider.delete(self.bank_id_for(user_id), document_id)
        except Exception as exc:
            logger.warning("[Memory Delete] 调用失败：%s", exc)
            return False

    def close(self) -> None:
        """释放底层客户端资源。"""
        self._provider.close()
