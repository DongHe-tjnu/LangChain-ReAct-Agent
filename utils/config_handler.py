"""
yml
k : V
不参与用户问答逻辑，只读取 config 目录下的 yml 文件变成python字典，供其他文件使用
"""
import yaml
from utils.path_tool import get_abs_path#读取 config 目录下的 yml 文件

#chat_model_name: qwen3-max embedding_model_name: text-embedding-v4
def load_rag_config(config_path: str = get_abs_path("config/rag.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_chroma_config(config_path: str = get_abs_path("config/chroma.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

#system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])执行这句把yml里面的内容变成字典然后通过prompts_conf["main_prompt_path"]拿到键对应的值
def load_prompts_config(config_path: str = get_abs_path("config/prompts.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str = get_abs_path("config/agent.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_memory_config(config_path: str = get_abs_path("config/memory.yml"), encoding: str = "utf-8"):
    """加载长期记忆模块配置。"""
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
agent_conf = load_agent_config()
prompts_conf = load_prompts_config()
memory_conf = load_memory_config()

if __name__ == '__main__':
    print(agent_conf["chat_model_name"])  # 输出对应配置
