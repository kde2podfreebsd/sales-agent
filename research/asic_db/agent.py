import json, os, re
from typing import Dict, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.agents import AgentOutputParser

from gsheet import GoogleSheets

load_dotenv()
gs = GoogleSheets()
headers: List[str] = list(gs.col_letter.keys())
session_mapping: Dict[str, int] = {}

def list_all_products(_: str = "") -> str:
    """JSON {row: название} всей колонки A."""
    return json.dumps(gs._get_col_a(), ensure_ascii=False)

class MapInput(BaseModel):
    mapping: Dict[str, int] = Field(..., description="index→row")

def _store(mapping: Dict[str, int]) -> str:
    if mapping:
        keys_int  = all(k.isdigit() for k in mapping.keys())
        vals_int  = all(isinstance(v, int) for v in mapping.values())
        if keys_int and vals_int:
            keys = list(map(int, mapping.keys()))
            vals = list(mapping.values())
            if min(keys) > 7 and max(vals) <= 7:
                mapping = {str(v): k for k, v in mapping.items()} 

    session_mapping.clear()
    session_mapping.update(mapping)
    return "stored"

store_mapping_tool = StructuredTool.from_function(
    name="store_mapping",
    func=_store,
    args_schema=MapInput,
    description="Сохраняет index→row для текущего списка.",
)

class IdsInput(BaseModel):
    indices: List[int] = Field(..., description="Номера из списка (1,2,3)")

def _fields_by_index(indices: List[int]) -> str:
    rows = [session_mapping.get(str(i)) for i in indices if str(i) in session_mapping]
    payload = {str(r): gs.get_product_fields(r, headers) for r in rows if r}
    return json.dumps(payload, ensure_ascii=False)

get_fields_tool = StructuredTool.from_function(
    name="get_fields_by_index",
    func=_fields_by_index,
    args_schema=IdsInput,
    description='{"indices":[1,3]} → JSON {row:{header:value}}',
)

tools = [
    Tool("list_all_products", list_all_products,
        description='Верни JSON {"row": "название"}. Аргумент: ""'),
    store_mapping_tool,
    get_fields_tool,
]

ACTION_RE = re.compile(r"^Action:\s*([^\n]+)\nAction Input:\s*(.+)", re.S | re.M)
class FixingOutputParser(AgentOutputParser):
    def parse(self, text: str):
        m = ACTION_RE.search(text)
        if m:
            tool, raw = m.group(1).strip(), m.group(2).strip()
            try:
                tool_input = json.loads(raw)
            except json.JSONDecodeError:
                tool_input = raw.strip('"')
            return AgentAction(tool, tool_input, text)
        if re.search(r"^Final Answer:", text, re.M):
            return AgentFinish({"output": text.split("Final Answer:", 1)[1].strip()}, text)
        fixed = "Thought: I now know the final answer\nFinal Answer: " + text.strip()
        return AgentFinish({"output": text.strip()}, fixed)
    @property
    def _type(self): return "fixing_output_parser"

prompt_template = """
Ты — вежливый продавец ASIC-оборудования (цены в USDT).

Инструменты:
• list_all_products — без аргумента, даёт JSON {{row: название}}  
• store_mapping — передай {{ "mapping":{{"1":13,"2":14}} }} → index→row  
    (если перепутаешь порядок, система сама перевернёт).  
• get_fields_by_index — {{ "indices":[1,3] }} → JSON с полями.

Формат ReAct:
Question: …
Thought: …
Action: list_all_products
Action Input: ""
Observation: …

Thought: …
Action: store_mapping
Action Input: {{ "mapping":{{"1":13,"2":14}} }}
Observation: stored

Thought: I now know the final answer
Final Answer: 1) Bitmain … 2) …

Алгоритм:
1. list_all_products → выбери ≤7 подходящих, покажи клиенту список
    только с номерами (без row). Затем вызови store_mapping.
2. Клиент указывает «1» или «1,3» и т.п.  
    → get_fields_by_index → короткий ответ (цена, мощность…).

{tools}

Имена инструментов: {tool_names}

Предыдущий диалог:
{chat_history}

Begin!

Question: {input}
Thought:{agent_scratchpad}
Отвечай строго на русском. Можно использовать крипто и майнинг сленг. 
""".strip()

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["input", "tools", "tool_names", "agent_scratchpad", "chat_history"],
)

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2,
                openai_api_key=os.getenv("OPENAI_API_KEY"))
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = AgentExecutor(
    agent=create_react_agent(llm, tools, prompt,
                            output_parser=FixingOutputParser()),
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=15,
    handle_parsing_errors=True,
)

def respond(q: str) -> str:
    return agent.invoke({"input": q})["output"]

if __name__ == "__main__":
    print("💬 Агент по продаже ASIC готов. Напиши «выход» для завершения.\n")
    while True:
        try:
            msg = input("🧑 Ты: ").strip()
            if msg.lower() in {"выход", "exit", "quit"}:
                print("👋 До связи!")
                break
            print(f"🤖 Агент: {respond(msg)}\n")
        except KeyboardInterrupt:
            print("\n👋 До встречи!")
            break
        except Exception as e:
            print(f"⚠️ Ошибка: {e}\n")
