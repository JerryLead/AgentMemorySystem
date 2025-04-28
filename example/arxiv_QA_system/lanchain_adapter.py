import uuid
from typing import Dict, List, Optional, Tuple, Any
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_community.chat_models import ChatSparkLLM
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage,Document
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from semantic_data_structure.semantic_simple_graph import SemanticSimpleGraph
from semantic_data_structure.semantic_map import SemanticMap
from langchain.embeddings.base import Embeddings
from datetime import datetime


class LangChainMemoryAdapter(SemanticSimpleGraph):
    """
    实现与LangChain兼容的对话记忆存储接口，支持：
    1. 对话消息的存储与语义检索
    2. 对话上下文的关联管理
    3. 元数据存储（时间戳、用户ID等）
    """
    
    def __init__(self, semantic_map=None):
        super().__init__(semantic_map)
        self.message_index = {}  # {message_id: key} 快速查找映射
        
    def _generate_message_id(self) -> str:
        return str(uuid.uuid4())

    def add_message(
        self,
        content: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        parent_id: Optional[str] = None,
        relation_type: str = "reply"
    ) -> str:
        """
        添加对话消息，返回message_id
        """
        message_id = self._generate_message_id()
        timestamp = datetime.now().isoformat()
        
        # 构建存储值（包含内容和元数据）
        value = {
            "content": content,
            "metadata": {
                "timestamp": timestamp,
                "user_id": user_id,
                **(metadata or {})
            }
        }
        
        # 插入语义图（使用content生成embedding）
        super().add_node(
            key=message_id,
            value=value,
            parent_keys=[parent_id] if parent_id else None,
            parent_relation=relation_type,
            text_for_embedding=content
        )
        
        self.message_index[message_id] = message_id
        return message_id

    def get_messages(
        self,
        message_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        获取原始消息记录
        """
        results = []
        count = 0
        
        for key in self.semantic_map.data:
            if count >= limit:
                break
                
            msg_id = key[0]
            value = key[1]
            
            # 过滤条件
            if message_ids and msg_id not in message_ids:
                continue
            if user_id and value["metadata"].get("user_id") != user_id:
                continue
                
            results.append({
                "id": msg_id,
                **value
            })
            count += 1
            
        return results

    def search_messages(
        self,
        query: str,
        k: int = 5,
        filter_condition: Optional[Dict] = None
    ) -> List[Dict]:
        """
        语义搜索消息
        """
        # 先进行语义检索
        similar_nodes = self.retrieve_similar_nodes(query, k=k)
        
        # 应用过滤条件
        filtered = []
        for node in similar_nodes:
            metadata = node["value"]["metadata"]
            if self._check_filter(metadata, filter_condition):
                filtered.append({
                    "id": node["key"],
                    "content": node["value"]["content"],
                    "metadata": metadata,
                    "distance": node["distance"]
                })
                
        return filtered

    def _check_filter(self, metadata: Dict, filters: Optional[Dict]) -> bool:
        if not filters:
            return True
            
        for k, v in filters.items():
            if metadata.get(k) != v:
                return False
        return True

    def get_conversation_thread(self, message_id: str) -> List[Dict]:
        """
        获取完整的对话线程
        """
        thread = []
        current_id = message_id
        
        # 向上回溯父消息
        while current_id:
            node = self.semantic_map.data.get(current_id)
            if not node:
                break
                
            thread.insert(0, {
                "id": current_id,
                "content": node["value"]["content"],
                "metadata": node["value"]["metadata"]
            })
            
            # 获取父消息（假设单亲结构）
            parents = self.get_parents(current_id)
            current_id = list(parents.keys())[0] if parents else None
            
        return thread

    def delete_message(self, message_id: str):
        """
        删除消息及其关联关系
        """
        if message_id in self.message_index:
            super().delete_node(message_id)
            del self.message_index[message_id]

    # 实现LangChain的Memory接口
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        """ 保存对话上下文 """
        input_str = "\n".join([f"{k}: {v}" for k, v in inputs.items()])
        output_str = "\n".join([f"{k}: {v}" for k, v in outputs.items()])
        
        # 创建关联消息
        parent_id = self._get_last_message_id()
        self.add_message(input_str, relation_type="input")
        self.add_message(output_str, parent_id=parent_id, relation_type="output")

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """ 加载记忆变量 """
        query = inputs.get("query", "")
        related = self.search_messages(query, k=3)
        return {"history": "\n".join([msg["content"] for msg in related])}

    def _get_last_message_id(self) -> Optional[str]:
        """ 获取最后一条消息ID """
        if not self.message_index:
            return None
        return next(reversed(self.message_index))
    


# if __name__ == "__main__":
#     # 初始化记忆系统
#     memory = LangChainMemoryAdapter()

#     # 存储对话
#     input_msg = "What's the capital of France?"
#     output_msg = "The capital of France is Paris."

#     memory.save_context(
#         {"user": input_msg},
#         {"assistant": output_msg}
#     )

#     # 语义搜索
#     results = memory.search_messages("European capitals", k=2)
#     print(results)
#     # 获取对话历史
#     history = memory.load_memory_variables({})

SPARKAI_URL = 'wss://spark-api.xf-yun.com/v1.1/chat'
SPARKAI_APP_ID = "82d11d66"
SPARKAI_API_SECRET = "MGFiNzk5YWYwYzg5NjUzYmExMzk1MDZi"
SPARKAI_API_KEY = "bde73051654bc4bc60a6d9d86f215aa2"
SPARKAI_DOMAIN = 'lite'
llm = ChatSparkLLM(
    spark_api_url=SPARKAI_URL,
        spark_app_id=SPARKAI_APP_ID,
        spark_api_key=SPARKAI_API_KEY,
        spark_api_secret=SPARKAI_API_SECRET,
        spark_llm_domain=SPARKAI_DOMAIN,
        streaming=False
)
memory = LangChainMemoryAdapter()
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a helpful travel agent AI. Use the provided context to personalize your responses and remember user preferences and past interactions. 
    Provide travel recommendations, itinerary suggestions, and answer questions about destinations. 
    If you don't have specific information, you can make general suggestions based on common travel knowledge."""),
    MessagesPlaceholder(variable_name="context"),
    HumanMessage(content="{input}")
])


def retrieve_context(query: str, user_id: str) -> List[Dict]:
    """Retrieve relevant context from LangChainMemoryAdapter"""
    memories = memory.search_messages(query, k=5, filter_condition={"user_id": user_id})
    serialized_memories = ' '.join([mem["content"] for mem in memories])
    context = [
        {
            "role": "system",
            "content": f"Relevant information: {serialized_memories}"
        },
        {
            "role": "user",
            "content": query
        }
    ]
    return context


def generate_response(input: str, context: List[Dict]) -> str:
    """Generate a response using the language model"""
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "input": input
    })
    return response.content


def save_interaction(user_id: str, user_input: str, assistant_response: str):
    """Save the interaction to LangChainMemoryAdapter"""
    interaction = [
        {
            "role": "user",
            "content": user_input
        },
        {
            "role": "assistant",
            "content": assistant_response
        }
    ]
    # 保存input
    memory.add_message(
        content=user_input,
        user_id=user_id,
        metadata={"type": "user_input"}
    )
    # 保存response
    memory.add_message(
        content=assistant_response,
        user_id=user_id,
        parent_id=memory._get_last_message_id(),  
        metadata={"type": "ai_response"}
    )


def chat_turn(user_input: str, user_id: str) -> str:
    # Retrieve context
    context = retrieve_context(user_input, user_id)

    # Generate response
    response = generate_response(user_input, context)

    # Save interaction
    save_interaction(user_id, user_input, response)

    return response


if __name__ == "__main__":
    print("Welcome to your personal Travel Agent Planner! How can I assist you with your travel plans today?")
    user_id = "john"

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Travel Agent: Thank you for using our travel planning service. Have a great trip!")
            break

        response = chat_turn(user_input, user_id)
        print(f"Travel Agent: {response}")