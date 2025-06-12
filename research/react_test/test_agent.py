import os
import json
import re

from gsheet import GoogleSheets
from dotenv import load_dotenv

from langchain_openai import OpenAI
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory

load_dotenv()
gs = GoogleSheets()

openai_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(
    model_name="gpt-4o-mini",
    temperature=0.0,
    openai_api_key=openai_key
)

def get_asic_info(user_query: str) -> int:
    products = gs.get_products_name()
    prompt = (
        f"Вот список асик устройств, которые есть в наличии: {products}\n"
        "Продукты представлены в виде hashmap — <ID>: <полное название модели>.\n"
        "На основании запроса пользователя выбери ровно один ID модели ASIC.\n"
        f"Запрос пользователя: {user_query}\n"
        "Верни строго JSON в формате:\n"
        '{"id": int}\n'
        "Где вместо int — ID выбранного устройства. Ничего лишнего."
        'Если подходящего асика нет, то верни {"id": "None"}'
    )

    llm_response = llm.invoke(prompt)

    match = re.search(r"\{.*\}", llm_response, re.DOTALL)
    if not match:
        raise ValueError(f"Не удалось найти JSON в ответе LLM: {llm_response!r}")
    json_str = match.group(0).replace("'", '"')

    json_id = json.loads(json_str)

    if json_id['id'] == 'None':
        return "Нет такого асик устройства"

    product_info = gs.get_product_info(row_number=int(json_id['id']))

    return product_info

# --- Инструмент #1: список моделей ---
# products_list_tool = Tool(
#     name="products list",
#     func=lambda _: gs.get_products_name(),
#     description=(
#         "Возвращает доступные модели майнеров"
#     ),
# )

# product_info_tool = Tool(
#     name="product info",
#     func= get_asic_info,
#     description=(
#         "Универсальная функция: по тексту запроса находит id нужного ASIC,"
#         "достаёт его полную информацию из Google Sheets и возвращает из неё"
#         "лишь ту часть, которую спрашивал пользователь."
#     ),
# )

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# agent = initialize_agent(
#     tools=[products_list_tool, product_info_tool],
#     llm=llm,
#     agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#     memory=memory,
#     verbose=True,
# )

print(get_asic_info("у вас есть вотсмайнер m50s?"))

#while True:
    # user = input("\nВы: ")
    # if user.lower() in {"exit", "quit"}: break
    # print("🤖:", agent.invoke({"input": user})["output"])