"""
FixingOutputParser
──────────────────
Общий парсер ReAct-вывода для всех sub-агентов и оркестратора.

•  Если встречается маркер **Final Answer:** – считаем, что агент закончил
    работу и возвращаем только то, что после маркера.
•  Иначе пытаемся извлечь пару **Action / Action Input**.
•  Если ни того ни другого нет – воспринимаем всё как финальный ответ.
"""

from __future__ import annotations

import json
import re
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.agents import AgentOutputParser

ACTION_RE = re.compile(
    r"^Action:\s*([^\n]+)\nAction Input:\s*(.+)", re.S | re.M
)


class FixingOutputParser(AgentOutputParser):
    """Универсальный output-parser для ReAct-агентов."""

    def parse(self, text: str) -> AgentAction | AgentFinish:
        if "Final Answer:" in text:
            final = text.split("Final Answer:", 1)[1].strip()
            return AgentFinish(return_values={"output": final}, log=text)

        m = ACTION_RE.search(text)
        if m:
            tool_name = m.group(1).strip()
            raw_input = m.group(2).strip()

            try:
                tool_input = json.loads(raw_input)
            except json.JSONDecodeError:
                tool_input = raw_input.strip().strip('"')

            return AgentAction(
                tool=tool_name,
                tool_input=tool_input,
                log=text,
            )

        return AgentFinish(return_values={"output": text.strip()}, log=text)

    @property
    def _type(self) -> str:  
        return "fixing_output_parser"
