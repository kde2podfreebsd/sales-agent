"""
Intent-Agent
────────────
•  Одним вызовом определяет, *что* хочет клиент и *какие сущности* уже
   содержатся в реплике.  
•  Всегда отвечает **строго одним JSON-объектом** без комментариев и
   форматирования.

Стандартизированный формат ответа:
{
  "intent": "<один из INTENTS>",
  "entities": {
      // см. обязательные и опциональные поля в INTENT_SPECS
  }
}
"""

from __future__ import annotations
import json, re
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import CombinedMemory
from langchain.agents import AgentExecutor


# ╔════════════════════════════════════════════════════════╗
#                       INTENT-СПЕЦИФИКАЦИИ
# ╚════════════════════════════════════════════════════════╝
INTENT_SPECS: Dict[str, str] = {
    "catalog_query": """
📦  catalog_query  
    •  Клиент запрашивает ассортимент, цену, характеристики, наличие,
       сравнение моделей, брендов или серий.  
    •  Ключевые триггеры: «какие есть», «почём», «цена», «сколько стоит»,
       «наличие», «характеристики», «хэшрейт», «S19 или M30s?», «что дешевле».  
    •  Обязательные entities _(если встречаются в тексте)_: brand,
       model, hash_rate, budget, condition.
""",
    "objection": """
🚧  objection  
    •  Клиент высказывает сомнения, страхи, негатив: «дорого», «шумит»,
       «окупится?», «гарантия», «расход по розетке», «сгорит ли» и т.п.  
    •  Обязательный entity: objection_type — одно из
       [price, noise, roi, warranty, power, heat, other].
""",
    "presentation": """
🎤  presentation  
    •  Запрос о компании, услугах, логистике, сервисе: «кто вы такие»,
       «условия поставки», «как работаете», «монтаж/настройка», «доверяю?»  
    •  Обязательных entities нет (можно пустой объект).
""",
    "schedule_call": """
📞  schedule_call  
    •  Клиент предлагает / соглашается на звонок/встречу *или* оставляет
       контакт (телефон, @telegram, e-mail).  
    •  Entities: phone, telegram, email, preferred_time (если указано).
""",
    "greeting": """
🙏  greeting  
    •  Приветствие, благодарность, прощание, small-talk
       («привет», «спасибо», «добрый вечер», «пока»).  
    •  Обычно ответ без использования инструментов.
""",
    "other": """
❓  other  
    •  Всё, что не попало в категории выше.
"""
}

INTENTS = list(INTENT_SPECS.keys())


# ╔════════════════════════════════════════════════════════╗
#                 PROMPT ДЕТЕКТОРА НАМЕРЕНИЙ
# ╚════════════════════════════════════════════════════════╝
SYSTEM_MSG = """
Ты — NLU-модуль продажного бота.  
Твоя задача: вернуть **один** JSON-объект ровно в следующем формате  
(никакого markdown, текста до или после):

{
  "intent": "<INTENT>",
  "entities": { ... }
}

1. Используй следующие INTENTS и их строгие определения:
{intent_definitions}

2. Определи entities по нужному списку для выбранного INTENT.
   •  phone  → любое 7-11-значное число, определи даже без «+».
   •  telegram → начинается с «@» или домен t.me/…
   •  brand / model  → слова вида Bitmain, Antminer S19 Pro, M30S++ …
   •  objection_type → price | noise | roi | warranty | power | heat | other
   •  hash_rate → число + «Th»/«TH»/«T»  (например 110 TH/s)
   •  budget → число + «₽» / «$» / «USDT»  (ставь raw строку)
   •  condition → новый | бу | восстановленный
   •  preferred_time → ISO-8601 YYYY-MM-DD HH:MM  (если клиент указал дату/время)

3. Если сущность не упоминается, просто не включай её в entities.

4. **Валидация ответа**:
   •  intent ∈ {intents}.  
   •  entities - объект (может быть пустым).

Верни JSON однострочно без пробелов и переносов.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG.strip().replace("{intent_definitions}",
                                            "\n".join(INTENT_SPECS.values()))),
        ("human", "{input}"),
    ]
)


# ╔════════════════════════════════════════════════════════╗
#                     AGENT-СТРОИТЕЛЬ
# ╚════════════════════════════════════════════════════════╝
def build(mem: CombinedMemory, llm: ChatOpenAI) -> AgentExecutor:
    """Возвращает Executor для определения интентов."""
    chain = LLMChain(llm=llm, prompt=PROMPT, memory=mem, verbose=False)
    return AgentExecutor(agent=chain, tools=[], memory=mem, verbose=False)


# ╔════════════════════════════════════════════════════════╗
#                СИНХРОННЫЙ ВЫЗОВ ВНЕ ЧЕЙНА
# ╚════════════════════════════════════════════════════════╝
def classify(message: str, exe: AgentExecutor):
    """
    Вспомогательная функция: быстро получить dict {intent, entities}
    без try/except-шуму в оркестраторе.
    """
    raw = exe.invoke({"input": message})["output"]
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "intent" in data and "entities" in data:
            return data
    except Exception: 
        pass
    return {"intent": "other", "entities": {}}
