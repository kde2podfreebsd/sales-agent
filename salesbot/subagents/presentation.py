from langchain.memory import CombinedMemory
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Ты делаешь короткую презентацию компании: "
            "○ 8 лет на рынке  "
            "○ Гарантия 12 мес  "
            "○ Склад в Москве, отправка в день оплаты  "
            "○ Помогаем настроить пул  "
            "Заканчивай открытым вопросом «Перейдём к выбору модели?»"),
    ("human", "{input}")
])

def build(shared: CombinedMemory, llm: ChatOpenAI) -> AgentExecutor:
    chain = LLMChain(llm=llm, prompt=PROMPT, memory=shared, verbose=True)
    return AgentExecutor(agent=chain, tools=[], memory=shared, verbose=True)
