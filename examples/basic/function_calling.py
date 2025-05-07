from openai import OpenAI

from swarm import Swarm, Agent

cclient = OpenAI(base_url="https://api.chatanywhere.tech/v1")

client = Swarm(client=cclient)


def get_weather(location) -> str:
    return "{'temp':67, 'unit':'F'}"


agent = Agent(
    name="Agent",
    model="gpt-4o-mini",
    instructions="You are a helpful agent.",
    functions=[get_weather],
)

messages = [{"role": "user", "content": "What's the weather in NYC?"}]

response = client.run(agent=agent, messages=messages,debug=True,)
print(response.messages[-1]["content"])
