import os
import json
import re
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory

from gsheet import GoogleSheets

load_dotenv()
gs = GoogleSheets()

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

def list_products(_: str) -> str:
    return json.dumps(gs.get_products_name(), ensure_ascii=False)


def fetch_product_info(product_id: str) -> str:
    try:
        return json.dumps(
            gs.get_product_info(row_number=int(product_id)),
            ensure_ascii=False,
        )
    except Exception:
        return "Нет такого asic устройства."


tools = [
    Tool(
        name="list_products",
        func=list_products,
        description=(
            "Используй, чтобы получить словарь <ID>: <название> всех ASIC-устройств. "
            "Не принимает аргументов."
        ),
    ),
    Tool(
        name="get_product_info",
        func=fetch_product_info,
        description=(
            "Используй, когда выбран конкретный ID. "
            "Передай ID (строкой или числом) и получи подробную информацию об ASIC."
        ),
    ),
]
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  
    verbose=True,  
    memory=memory,
)

def get_asic_info(user_query: str) -> dict:
    """
    • Агент получает запрос пользователя.
    • Сам вызывает list_products → выбирает ID → get_product_info.
    • Возвращает JSON вида {"id": <ID|None>, "info": <данные|сообщение>}.
    """
    response = agent.run(
        "Твоя задача: подобрать один ASIC под запрос пользователя. "
        "Сначала вызови list_products, выбери подходящий ID, затем вызови get_product_info. "
        "В финале верни только JSON вида "
        '{"id": <id|None>, "info": <описание или "Нет такого asic устройства">}. '
        f"Запрос пользователя: {user_query}"
    )

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        raise ValueError(f"Не удалось найти JSON в ответе агента: {response!r}")

    return json.loads(match.group(0).replace("'", '"'))


get_asic_info("по сколько вотсмайнеры m60 на 164тх?")