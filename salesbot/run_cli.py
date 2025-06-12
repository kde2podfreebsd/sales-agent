from salesbot.orchestrator import build_orchestrator

agent = build_orchestrator()

print("💬 ASIC-бот v2. Пиши 'exit' для выхода")
while True:
    msg = input("🧑: ")
    if msg.lower() in {"exit","quit"}:
        break
    try:
        print("🤖:", agent.invoke({"input": msg})["output"])
    except Exception as e:
        print("⚠️", e)
