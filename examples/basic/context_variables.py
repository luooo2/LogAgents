from openai import OpenAI

from swarm import Swarm, Agent

cclient = OpenAI(base_url="https://api.chatanywhere.tech/v1")

client = Swarm(client=cclient)


def instructions(context_variables):
    name = context_variables.get("name", "User")
    return f"You are a helpful agent. Greet the user by name ({name})."


def print_account_details(context_variables: dict):
    user_id = context_variables.get("user_id", None)
    name = context_variables.get("name", None)
    print(f"Account Details: {name} {user_id}")
    return "Success"


agent = Agent(
    name="Agent",
    model="gpt-4o-mini",
    instructions=instructions,
    functions=[print_account_details],
)

context_variables = {"name": "Luooo", "user_id": 123}

# response = client.run(
#     messages=[{"role": "user", "content": "Hi! Please tell me about the ai."}],
#     agent=agent,
#     context_variables=context_variables,
# )
# print(response.messages[-1]["content"])

response = client.run(
    messages=[{"role": "user", "content": "Print my account details!"}],
    agent=agent,
    context_variables=context_variables,
    debug=True,
)
print(response.messages[-1]["content"])
