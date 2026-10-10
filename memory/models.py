"""长期记忆模块使用的领域模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class MemoryType(str, Enum):
    """业务层支持的长期记忆类型。"""

    FACT = "Fact"
    PREFERENCE = "Preference"
    PROJECT = "Project"
    EXPERIENCE = "Experience"
    PROFILE = "Profile"
    TASK = "Task"


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """与具体 Memory 服务无关的单条长期记忆。"""

    content: str
    memory_type: MemoryType = MemoryType.FACT
    document_id: str | None = None
    memory_id: str | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecallResult:
    """Recall 与可选 Reflection 的统一返回值。"""

    memories: tuple[MemoryRecord, ...] = ()
    reflection: str | None = None

    @property
    def is_empty(self) -> bool:
        """返回结果是否不包含任何可用上下文。"""
        return not self.memories and not self.reflection

    def to_prompt_context(self) -> str:
        """转换为可追加到 System Prompt 的纯文本上下文。"""
        sections: list[str] = []
        if self.memories:
            memory_lines = "\n".join(f"- {item.content}" for item in self.memories)
            sections.append(f"Relevant Memories:\n{memory_lines}")
        if self.reflection:
            sections.append(f"Memory Reflection:\n{self.reflection}")
        return "\n\n".join(sections)
