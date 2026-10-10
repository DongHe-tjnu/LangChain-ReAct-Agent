import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils.config_handler import rag_conf


class ModelProvider(str, Enum):
    """项目支持的模型服务供应商。"""

    SILICONFLOW = "siliconflow"
    DASHSCOPE = "dashscope"


def _get_provider() -> ModelProvider:
    """读取模型供应商，允许通过环境变量覆盖 YAML 配置。"""
    configured_provider = os.getenv(
        "MODEL_PROVIDER",
        str(rag_conf.get("model_provider", ModelProvider.SILICONFLOW.value)),
    ).lower()
    try:
        return ModelProvider(configured_provider)
    except ValueError as exc:
        supported = ", ".join(provider.value for provider in ModelProvider)
        raise ValueError(
            f"不支持的模型供应商 {configured_provider!r}，可选值：{supported}"
        ) from exc


def _require_env(name: str) -> str:
    """返回必需的环境变量，并在缺失时提供可操作的错误信息。"""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少环境变量 {name}，请先配置模型服务 API Key。")
    return value


# 创建 chat_model：给 Agent 和 RAG 总结用；创建 embed_model：给 Chroma 向量库用。
class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    """根据配置创建聊天模型。"""

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        provider = _get_provider()
        if provider is ModelProvider.SILICONFLOW:
            return ChatOpenAI(
                model=rag_conf["chat_model_name"],
                api_key=_require_env("SILICONFLOW_API_KEY"),
                base_url=rag_conf["siliconflow_base_url"],
            )
        return ChatTongyi(model=rag_conf["chat_model_name"])


class EmbeddingsFactory(BaseModelFactory):
    """根据配置创建文本向量模型。"""

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        provider = _get_provider()
        if provider is ModelProvider.SILICONFLOW:
            return OpenAIEmbeddings(
                model=rag_conf["embedding_model_name"],
                api_key=_require_env("SILICONFLOW_API_KEY"),
                base_url=rag_conf["siliconflow_base_url"],
            )
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = ChatModelFactory().generator()

embed_model = EmbeddingsFactory().generator()
