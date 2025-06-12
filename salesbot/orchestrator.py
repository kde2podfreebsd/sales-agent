"""
Sales-Orchestrator
──────────────────
•  Определяет intent  → вызывает нужный sub-agent.
•  Для catalog_agent формирует отдельную «короткую» CombinedMemory
    (последние 3 реплики + все сущности) и передаёт в sub-агенте.
•  Catalog-Agent пользуется Redis-инструментами list_all_products / store_mapping /
    get_fields_by_index согласно ReAct-алгоритму.
"""

from __future__ import annotations

# ── std & 3-rd ───────────────────────────────────────────────────────
from typing import List
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain.memory import (
    CombinedMemory,
    ConversationBufferMemory,
    ConversationEntityMemory,
)

# ── local ────────────────────────────────────────────────────────────
from salesbot.memory import build_shared
from salesbot.tools_catalog import catalog_tools
from salesbot.subagents import catalog, objections, presentation, schedule_call, intent
from salesbot.subagents.output_parser import FixingOutputParser


# ═════════════════════════════════════════════════════════════════════
# helpers
# ═════════════════════════════════════════════════════════════════════
def _make_catalog_memory(full_mem: CombinedMemory, llm: ChatOpenAI, window: int = 3) -> CombinedMemory:
    """
    Создаём временную CombinedMemory для Catalog-Agent:
    • берём последние <window> пользователь-бот реплик
    • + общую EntityMemory (name, phone, chosen_brand …)
    """
    buf_full: ConversationBufferMemory = next(
        m for m in full_mem.memories if isinstance(m, ConversationBufferMemory)
    )
    ent_full: ConversationEntityMemory = next(
        m for m in full_mem.memories if isinstance(m, ConversationEntityMemory)
    )

    win_buf = ConversationBufferMemory(
        return_messages=True, memory_key="chat_history"
    )
    last_msgs = buf_full.chat_memory.messages[-window * 2 :]
    for msg in last_msgs:
        win_buf.chat_memory.add_message(msg)

    return CombinedMemory(memories=[win_buf, ent_full])


def _wrap_subagent(name: str, exe: AgentExecutor, descr: str) -> Tool:
    """Обычный обёртка-Tool для «статических» sub-агентов."""
    return Tool(
        name=name,
        description=descr,
        func=lambda q: exe.invoke({"input": q})["output"],
    )


def _wrap_catalog(memory: CombinedMemory, llm: ChatOpenAI) -> Tool:
    """
    Специальный wrapper для Catalog-Agent.
    Каждый вызов:
    1. создаёт контекст-память с узким окном
    2. строит новый Executor (легко, он lightweight)
    3. передаёт tools/tool_names в prompt (для {tools} placeholder)
    """
    cat_tools = catalog_tools()
    tool_names = [t.name for t in cat_tools]

    def _call(q: str) -> str:
        cat_mem = _make_catalog_memory(memory, llm)
        cat_exec = catalog.build(cat_mem, llm)

        result = cat_exec.invoke(
            {
                "input": q,
                "tools": cat_tools,
                "tool_names": tool_names,
            }
        )
        return result["output"]

    return Tool(
        name="catalog_agent",
        description="Отвечает на вопросы о моделях, ценах, брендах ASIC",
        func=_call,
    )


# ═════════════════════════════════════════════════════════════════════
# build orchestrator
# ═════════════════════════════════════════════════════════════════════
def build_orchestrator(session_id: str | None = None) -> AgentExecutor:
    # ── shared memory & llm ──────────────────────────────────────────
    shared_mem, llm = build_shared(session_id)

    # ── sub-executors (кроме каталога) ───────────────────────────────
    obj_exe = objections.build(shared_mem, llm)
    pre_exe = presentation.build(shared_mem, llm)
    call_exe = schedule_call.build(shared_mem, llm)
    nlp_exe = intent.build(shared_mem, llm)

    # ── Tools list ──────────────────────────────────────────────────
    tools: List[Tool] = [
        _wrap_catalog(shared_mem, llm),
        _wrap_subagent("objection_agent",   obj_exe, "Работа с возражениями"),
        _wrap_subagent("presentation_agent", pre_exe, "Презентация компании"),
        _wrap_subagent("schedule_call",     call_exe, "Согласование звонка"),
        _wrap_subagent("classify_intent",   nlp_exe, "Определяет intent и сущности"),
    ]

    # ── главный ReAct-Prompt ─────────────────────────────────────────
    TEMPLATE = """
Ты — старший продавец ASIC-оборудования.

1. Сначала вызови **classify_intent** чтобы определить намерение и сущности.
2. Дальнейшие действия:  
    • catalog_query  → catalog_agent  
    • objection      → objection_agent  
    • presentation   → presentation_agent  
    • schedule_call  → schedule_call  
    • greeting/other → ответь дружелюбно без инструментов
    3. В конце *объедини* вывод sub-агента с собственным вступлением/закрытием,
    но не повторяй данные дважды.
4. Если сущность (name / model / phone) уже есть в памяти — не переспрашивай.

{chat_history}
Question: {input}
Thought:{agent_scratchpad}
""".strip()

    prompt = PromptTemplate(
        template=TEMPLATE,
        input_variables=["chat_history", "input", "agent_scratchpad"],
    )

    # ── ReAct-agent ─────────────────────────────────────────────────
    react_agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        output_parser=FixingOutputParser(),
    )

    return AgentExecutor(
        agent=react_agent,
        tools=tools,
        memory=shared_mem,
        verbose=True,
        max_iterations=20,
    )
