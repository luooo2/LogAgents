from openai import OpenAI

from swarm import Swarm, Agent

cclient = OpenAI(base_url="https://api.chatanywhere.tech/v1")

client = Swarm(client=cclient)

agent = Agent(
    name="Agent",
    model="gpt-4o-mini",
    instructions="You are a helpful agent.",
)

messages = [{"role": "user", "content": "Hi! Can you tell me how to learn English?"}]
# 无法设置流式回答
response = client.run(agent=agent, messages=messages,debug=True,)

print(response.messages[-1]["content"])
