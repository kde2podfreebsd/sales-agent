
import os
import json
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv 

load_dotenv()

client = OpenAI()

def chat_with_gpt(
    system: str,
    user: str,
    model: str = "gpt-4",
    temperature: float = 0.0
) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
    )
    return resp.choices[0].message.content.strip()

class SlangNormalizerGPT:
    SYSTEM_PROMPT = """
Ты — ассистент, который нормализует сленговые выражения майнинговой тематики.
На вход получаешь текст от клиента на русском.
Твоя задача:
1) Расшифровать сленг ("розетка" -> "электропитание", "120тх" -> "120 TH/s" и т.п.).
2) Вернуть чистый, формальный текст на русском без объяснений, только результат нормализации.
"""

    @staticmethod
    def normalize(text: str) -> str:
        return chat_with_gpt(
            SlangNormalizerGPT.SYSTEM_PROMPT,
            f"Нормализуй этот текст:\n\"{text}\""
        )

class StageAnalyzerGPT:
    SYSTEM_PROMPT = """
Ты — ассистент, который по сообщению клиента определяет этап воронки продаж ASIC.
Возможные этапы (ключи): greeting, need_info, presentation, negotiation, closing, disqualification.
Верни JSON:
{
"stage": "<ключ этапа>",
"reason": "<короткое объяснение>"
}
"""

    @staticmethod
    def analyze(last_user: str) -> Dict[str, str]:
        out = chat_with_gpt(
            StageAnalyzerGPT.SYSTEM_PROMPT,
            f"Определи этап по сообщению:\n\"{last_user}\""
        )
        return json.loads(out)


class EntityExtractorGPT:
    SYSTEM_PROMPT = """
Ты — ассистент, который извлекает из сообщения клиента всю ключевую информацию для карточки.
Поля:
name, quantity, new_or_used, budget, placement,
electricity_price, location, experience, goals,
ready_to_buy, preferred_contact.
Верни JSON с этими полями. Если поле не упомянуто — значение null.
"""

    @staticmethod
    def extract(text: str) -> Dict[str, Any]:
        out = chat_with_gpt(
            EntityExtractorGPT.SYSTEM_PROMPT,
            f"Извлеки данные из сообщения:\n\"{text}\""
        )
        return json.loads(out)


class ClientProfileManager:
    def __init__(self):
        self.reset()

    def reset(self):
        fields = [
            "name", "quantity", "new_or_used", "budget", "placement",
            "electricity_price", "location", "experience", "goals",
            "ready_to_buy", "preferred_contact"
        ]
        self.profile: Dict[str, Any] = {f: None for f in fields}

    def update(self, data: Dict[str, Any]):
        for k, v in data.items():
            if v is not None:
                self.profile[k] = v

    def summary(self) -> str:
        parts = [f"{k}={v}" for k, v in self.profile.items() if v is not None]
        return "; ".join(parts) if parts else "профиль пуст"

class MemoryManager:
    def __init__(self):
        self.history: list = []

    def add(self, role: str, text: str):
        self.history.append({"role": role, "text": text})

    def get_recent(self, n: int = 10) -> list:
        return self.history[-n:]


ASIC_DB = [
    {"model": "Antminer S19j Pro", "hashrate": 120, "price": 310, "available": 3},
    {"model": "WhatsMiner M30S++", "hashrate": 112, "price": 300, "available": 5},
]

def search_asic_db(query: str) -> str:
    try:
        th = int(query.split("TH/s")[0].split()[-1])
    except:
        return "Неверный формат для поиска ASIC."
    res = [d for d in ASIC_DB if abs(d["hashrate"] - th) <= 5]
    if not res:
        return "Ничего не найдено."
    return "\n".join(
        f"{r['model']}: {r['hashrate']} TH/s, цена {r['price']} USDT, в наличии {r['available']} шт"
        for r in res
    )


PROFILE_FILE = "client_profile.json"
HISTORY_FILE = "dialog_history.json"

class SalesAgentPipelineGPT:
    def __init__(self):
        self.norm = SlangNormalizerGPT()
        self.stage = StageAnalyzerGPT()
        self.extractor = EntityExtractorGPT()
        self.profile = ClientProfileManager()
        self.memory = MemoryManager()

    def save_state(self):
        with open(PROFILE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.profile.profile, f, ensure_ascii=False, indent=2)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.memory.history, f, ensure_ascii=False, indent=2)

    def process(self, user_input: str) -> str:
        norm = self.norm.normalize(user_input)
        self.memory.add("user", norm)

        entities = self.extractor.extract(norm)
        self.profile.update(entities)

        stage_info = self.stage.analyze(norm)
        stage, reason = stage_info["stage"], stage_info["reason"]

        system_ctx = (
            f"Ты — AI-ассистент по продажам ASIC на Avito.\n"
            f"Текущая стадия: {stage}\n"
            f"Причина: {reason}\n"
            f"Карточка клиента: {self.profile.summary()}\n"
            "Если нужно — используй инструмент поиска ASIC.\n"
        )
        user_ctx = f"Пользователь: {norm}"

        if "TH/s" in norm:
            obs = search_asic_db(norm)
            system_ctx += f"\nObservation (из базы ASIC):\n{obs}\n"

        final_resp = chat_with_gpt(system_ctx, user_ctx)
        self.memory.add("agent", final_resp)
        self.save_state()
        return final_resp


if __name__ == "__main__":
    pipeline = SalesAgentPipelineGPT()
    print("=== AI-агент ASIC запущен. Введите сообщение.")
    print("Спецкоманды: /profile, /history, /exit")
    while True:
        msg = input("> ").strip()
        if not msg:
            continue
        if msg.lower() in ("/exit", "exit", "quit"):
            print("Выход.")
            break
        if msg == "/profile":
            print(json.dumps(pipeline.profile.profile, ensure_ascii=False, indent=2))
            continue
        if msg == "/history":
            print(json.dumps(pipeline.memory.history, ensure_ascii=False, indent=2))
            continue

        reply = pipeline.process(msg)
        print("AGENT:", reply)
        print()