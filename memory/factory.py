"""Memory 模块装配入口。"""

from __future__ import annotations

import os
from typing import Any, Mapping

from memory.contracts import NullMemoryProvider
from memory.memory_manager import MemoryManager
from utils.logger_handler import logger


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def create_memory_manager(config: Mapping[str, Any]) -> MemoryManager:
    """根据统一配置创建 MemoryManager，初始化失败时自动降级。"""
    enabled = _as_bool(os.getenv("ENABLE_MEMORY", config.get("ENABLE_MEMORY")), default=False)
    provider = NullMemoryProvider()
    provider_ready = False

    if enabled:
        try:
            # 延迟导入保证 ENABLE_MEMORY=false 时无需安装或初始化厂商 SDK。
            from memory.hindsight_client import HindsightMemoryClient

            provider = HindsightMemoryClient(
                base_url=os.getenv("HINDSIGHT_API_URL")
                or str(config.get("HINDSIGHT_API_URL", "http://localhost:8888")),
                api_key=os.getenv("HINDSIGHT_API_KEY") or config.get("HINDSIGHT_API_KEY"),
                timeout_seconds=float(config.get("REQUEST_TIMEOUT_SECONDS", 5.0)),
                recall_budget=str(config.get("RECALL_BUDGET", "mid")),
                recall_max_tokens=int(config.get("RECALL_MAX_TOKENS", 1000)),
                reflection_max_tokens=int(config.get("REFLECTION_MAX_TOKENS", 600)),
                max_attempts=int(config.get("MAX_ATTEMPTS", 1)),
            )
            provider_ready = True
        except Exception as exc:
            logger.warning("[Memory] 初始化失败，已切换为空实现：%s", exc)

    return MemoryManager(
        provider=provider,
        bank_id_prefix=str(config.get("BANK_ID_PREFIX", "langchain-react-agent")),
        reflection_enabled=_as_bool(config.get("ENABLE_REFLECTION"), default=True),
        search_limit=int(config.get("SEARCH_LIMIT", 20)),
        enabled=enabled and provider_ready,
    )
