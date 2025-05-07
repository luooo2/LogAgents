from openai import OpenAI

from swarm import Swarm, Agent

cclient = OpenAI(base_url="https://api.chatanywhere.tech/v1")

client = Swarm(client=cclient)

english_agent = Agent(
    name="English Agent",
    model="gpt-4o-mini",
    instructions="You only speak English.",
)

spanish_agent = Agent(
    name="Spanish Agent",
    instructions="You only speak Spanish.",
)


def transfer_to_spanish_agent():
    """Transfer spanish speaking users immediately."""
    return spanish_agent


english_agent.functions.append(transfer_to_spanish_agent)

messages = [{"role": "user", "content": "Hola. ¿Como estás?"}]
response = client.run(agent=english_agent, messages=messages)

print(response.messages[-1]["content"])
