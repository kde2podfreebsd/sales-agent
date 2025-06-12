import uuid
from typing import Tuple
from langchain.memory import CombinedMemory, ConversationBufferMemory
from langchain.memory import ConversationEntityMemory
from langchain_openai import ChatOpenAI

def build_shared(session_id: str | None = None) -> Tuple[CombinedMemory, ChatOpenAI]:
    if not session_id:
        session_id = str(uuid.uuid4())
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    buf = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    ent = ConversationEntityMemory(llm=llm, k=30)   # name, phone, model…
    return CombinedMemory(memories=[buf, ent]), llm
