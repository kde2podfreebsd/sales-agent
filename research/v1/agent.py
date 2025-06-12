# agent.py
import os
import re
import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.agents import AgentOutputParser

from v1.tools import get_tools

load_dotenv()

ACTION_RE = re.compile(r"^Action:\s*([^\n]+)\nAction Input:\s*(.+)", re.S | re.M)


class FixingOutputParser(AgentOutputParser):
    """
    1) Если встречает 'Final Answer:', возвращает только текст после него.
    2) Иначе пытается извлечь Action и Action Input.
    3) Во всех остальных случаях возвращает весь текст как финальный ответ.
    """
    def parse(self, text: str):
        if "Final Answer:" in text:
            final = text.split("Final Answer:", 1)[1].strip()
            return AgentFinish(return_values={"output": final}, log=text)

        m = ACTION_RE.search(text)
        if m:
            tool = m.group(1).strip()
            raw = m.group(2).strip()
            try:
                inp = json.loads(raw)
            except json.JSONDecodeError:
                inp = raw.strip().strip('"')
            return AgentAction(tool=tool, tool_input=inp, log=text)

        return AgentFinish(return_values={"output": text.strip()}, log=text)

    @property
    def _type(self) -> str:
        return "fixing_output_parser"


def build_agent() -> AgentExecutor:
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    tools = get_tools()
    tool_names = [t.name for t in tools]

    prompt_text = """
Ты — вежливый продавец ASIC-оборудования. Отвечай строго на русском, можно использовать крипто- и майнинг-сленг.

Инструменты:
• list_all_products — без аргументов, возвращает JSON, где ключи — номера строк в таблице, а значения — названия моделей.
• store_mapping — принимает JSON с полем mapping (словарь index→row) и сохраняет его в памяти.
• get_fields_by_index — принимает JSON с полем indices (список индексов) и возвращает JSON с полями моделей, включая «Производитель», «Линейка», «Состояние», «Цена продажи» и др.

Формат ReAct:
Question: <вопрос>
Thought: <мыслительный процесс>
Action: <имя инструмента>
Action Input: <JSON с параметрами>
Observation: <результат вызова инструмента>

Когда у тебя есть окончательный ответ:
Thought: I now know the final answer
Final Answer: <только текст ответа без мыслей>

Алгоритм для вопроса «какие ASIC в продаже?»:
1. Вызвать list_all_products без аргументов.
2. Сохранить весь список через store_mapping, передав mapping — соответствие порядковых индексов (1,2,3…) к номерам строк в таблице.
3. Вызвать get_fields_by_index для всех индексов, чтобы получить у каждой модели поля «Производитель», «Линейка» и «Состояние».
4. Если моделей больше 7:
    a) Thought: «много моделей, сгруппирую по производителю»
    b) Сгруппировать по полю «Производитель», посчитать количество моделей в каждой группе.
    c) Final Answer:  
    У нас в продаже следующие бренды ASIC:
    1. Bitmain (34 модели)  
    2. Whatsminer (34 модели)  
    3. Jasminer (4 модели)  
    …  
    Какой бренд вас интересует?
5. Если моделей 7 или меньше:
a) Thought: «моделей немного, покажу их список»
b) Final Answer:  
    1. [1] Bitmain Antminer S19 90 Th б/у  
    2. [2] Bitmain Antminer S19i 88Th  
    …  
    Напишите номера выбранных моделей.
6. После выбора бренда или списка:
    а) При выборе бренда — сгруппировать по полю «Линейка» и спросить, какую линейку выбрать.
    б) Затем спросить «бу или новые?» по полю «Состояние».
    в) Затем показать до 7 конкретных моделей и попросить номера.
    7. После получения списка номеров:
    a) Вызвать store_mapping с mapping для выбранных индексов.
    b) Вызвать get_fields_by_index с теми же индексами.
    c) Final Answer: выдать цену, потребление и характеристики выбранных моделей.

Не хардкодь бренды или серии — все группировки делай на основе полей из JSON, полученного через инструменты.

{tools}
Имена инструментов: {tool_names}
{chat_history}
Question: {input}
Thought:{agent_scratchpad}
""".strip()

    prompt = PromptTemplate(
        template=prompt_text,
        input_variables=["tools", "tool_names", "chat_history", "input", "agent_scratchpad"],
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return AgentExecutor(
        agent=create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt,
            output_parser=FixingOutputParser(),
        ),
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=25,
        handle_parsing_errors=True,
    )






