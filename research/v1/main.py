# main.py
import os
from dotenv import load_dotenv

from v1.agent import build_agent

load_dotenv()

def main():
    sessions: dict[str, Any] = {}

    print("💬 Агент по продаже ASIC готов. Напиши «выход» для завершения.")
    chat_id = "cli"  
    
    sessions[chat_id] = build_agent()

    while True:
        msg = input("🧑 Ты: ").strip()
        if msg.lower() in {"выход", "exit", "quit"}:
            print("👋 До связи!")
            break

        agent = sessions[chat_id]
        try:
            result = agent.invoke({"input": msg})
            print(f"🤖 Агент: {result['output']}\n")
        except Exception as e:
            print(f"⚠️ Ошибка в агенте: {e}\n")

if __name__ == "__main__":
    main()
