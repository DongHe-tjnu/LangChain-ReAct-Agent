"""Hindsight SDK Adapter。

本文件是项目中唯一直接依赖 Hindsight SDK 的位置。
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Iterable
from threading import Lock
from typing import Any, TypeVar

from hindsight_client import Hindsight

from memory.contracts import MemoryProvider
from memory.models import MemoryRecord, MemoryType

_T = TypeVar("_T")


class HindsightMemoryClient(MemoryProvider):
    """使用官方 Python SDK 实现的 Hindsight Memory Provider。"""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 5.0,
        recall_budget: str = "mid",
        recall_max_tokens: int = 1000,
        reflection_max_tokens: int = 600,
        max_attempts: int = 1,
    ) -> None:
        self._client_kwargs = {
            "base_url": base_url,
            "api_key": api_key,
            "timeout": timeout_seconds,
            "user_agent": "langchain-react-agent/1.0",
            "max_attempts": max_attempts,
        }
        self._recall_budget = recall_budget
        self._recall_max_tokens = recall_max_tokens
        self._reflection_max_tokens = reflection_max_tokens
        self._known_banks: set[str] = set()
        self._bank_lock = Lock()

    def _new_client(self) -> Hindsight:
        """为当前调用创建 SDK 会话，避免跨 Streamlit 线程复用异步连接。"""
        return Hindsight(**self._client_kwargs)

    def _ensure_bank(self, client: Hindsight, bank_id: str) -> None:
        """首次访问用户时自动创建其独立 Memory Bank。"""
        if bank_id in self._known_banks:
            return
        with self._bank_lock:
            if bank_id in self._known_banks:
                return
            client.create_bank(bank_id=bank_id)
            self._known_banks.add(bank_id)

    def retain(self, bank_id: str, memory: MemoryRecord) -> None:
        tags = list(memory.tags)
        tags.append(f"memory_type:{memory.memory_type.value.lower()}")
        metadata = {key: str(value) for key, value in memory.metadata.items()}
        with self._new_client() as client:
            self._ensure_bank(client, bank_id)
            client.retain(
                bank_id=bank_id,
                content=memory.content,
                context="The user is speaking about durable personal information.",
                document_id=memory.document_id,
                metadata=metadata,
                tags=tags,
                update_mode="replace",
                retain_async=True,
            )

    def recall(self, bank_id: str, query: str) -> tuple[MemoryRecord, ...]:
        with self._new_client() as client:
            self._ensure_bank(client, bank_id)
            response = client.recall(
                bank_id=bank_id,
                query=query,
                max_tokens=self._recall_max_tokens,
                budget=self._recall_budget,
                prefer_observations=True,
            )
        return self._convert_items(self._read(response, "results", ()))

    def reflect(self, bank_id: str, query: str) -> str | None:
        with self._new_client() as client:
            self._ensure_bank(client, bank_id)
            response = client.reflect(
                bank_id=bank_id,
                query=query,
                budget=self._recall_budget,
                max_tokens=self._reflection_max_tokens,
            )
        text = self._read(response, "text")
        return str(text).strip() if text else None

    def search(self, bank_id: str, query: str, limit: int) -> tuple[MemoryRecord, ...]:
        with self._new_client() as client:
            self._ensure_bank(client, bank_id)
            response = client.list_memories(
                bank_id=bank_id,
                search_query=query,
                limit=limit,
                state="valid",
            )
        items = self._read(response, "items", None)
        if items is None:
            items = self._read(response, "memories", ())
        return self._convert_items(items)

    def delete(self, bank_id: str, document_id: str) -> bool:
        with self._new_client() as client:
            self._ensure_bank(client, bank_id)
            operation = client.documents.delete_document(bank_id, document_id)
            response = self._run_coroutine(operation)
        success = self._read(response, "success", True)
        return bool(success)

    def close(self) -> None:
        """SDK 会话均按调用关闭，此处保留接口兼容性。"""

    @classmethod
    def _convert_items(cls, items: Iterable[Any]) -> tuple[MemoryRecord, ...]:
        records: list[MemoryRecord] = []
        for item in items:
            content = cls._read(item, "text", None) or cls._read(item, "content", None)
            if not content:
                continue
            records.append(
                MemoryRecord(
                    content=str(content),
                    memory_type=cls._memory_type_from_tags(cls._read(item, "tags", ())),
                    document_id=cls._read(item, "document_id", None),
                    memory_id=cls._read(item, "id", None),
                    tags=tuple(cls._read(item, "tags", ()) or ()),
                    metadata=cls._read(item, "metadata", {}) or {},
                )
            )
        return tuple(records)

    @staticmethod
    def _read(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    @staticmethod
    def _memory_type_from_tags(tags: Iterable[str] | None) -> MemoryType:
        for tag in tags or ():
            if not tag.startswith("memory_type:"):
                continue
            value = tag.partition(":")[2]
            for memory_type in MemoryType:
                if memory_type.value.lower() == value.lower():
                    return memory_type
        return MemoryType.FACT

    @staticmethod
    def _run_coroutine(coroutine: Coroutine[Any, Any, _T]) -> _T:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coroutine)
