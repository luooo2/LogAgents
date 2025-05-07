from openai import OpenAI

from swarm import Swarm, Agent

cclient = OpenAI(base_url="https://api.chatanywhere.tech/v1")

client = Swarm(client=cclient)

my_agent = Agent(
    name="Agent",
    model="gpt-4o-mini",
    instructions="You are a helpful agent.",
)


def pretty_print_messages(messages):
    for message in messages:
        if message["content"] is None:
            continue
        print(f"{message['sender']}: {message['content']}")


messages = []
agent = my_agent
while True:
    user_input = input("> ")
    messages.append({"role": "user", "content": user_input})

    response = client.run(agent=agent, messages=messages)
    messages = response.messages
    agent = response.agent
    pretty_print_messages(messages)
