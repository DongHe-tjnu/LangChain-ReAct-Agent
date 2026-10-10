"""Memory Provider 的抽象边界。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from memory.models import MemoryRecord


class MemoryProvider(ABC):
    """长期记忆服务统一接口，业务层不得依赖具体厂商 SDK。"""

    @abstractmethod
    def retain(self, bank_id: str, memory: MemoryRecord) -> None:
        """保存一条已经提炼的长期记忆。"""

    @abstractmethod
    def recall(self, bank_id: str, query: str) -> tuple[MemoryRecord, ...]:
        """语义召回与当前问题相关的长期记忆。"""

    @abstractmethod
    def reflect(self, bank_id: str, query: str) -> str | None:
        """基于长期记忆生成综合反思结果。"""

    @abstractmethod
    def search(self, bank_id: str, query: str, limit: int) -> tuple[MemoryRecord, ...]:
        """按文本条件搜索记忆。"""

    @abstractmethod
    def delete(self, bank_id: str, document_id: str) -> bool:
        """删除一次 Retain 对应的文档及其派生记忆。"""

    def close(self) -> None:
        """释放 Provider 持有的资源。"""


class NullMemoryProvider(MemoryProvider):
    """关闭 Memory 时使用的空实现。"""

    def retain(self, bank_id: str, memory: MemoryRecord) -> None:
        return None

    def recall(self, bank_id: str, query: str) -> tuple[MemoryRecord, ...]:
        return ()

    def reflect(self, bank_id: str, query: str) -> str | None:
        return None

    def search(self, bank_id: str, query: str, limit: int) -> tuple[MemoryRecord, ...]:
        return ()

    def delete(self, bank_id: str, document_id: str) -> bool:
        return False
