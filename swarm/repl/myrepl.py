# 该代码文件用于实现用户与智能体的交互
import json

from openai import OpenAI

from swarm import Swarm

# 处理并格式化流式响应，将内容实时输出到终端，输出发送者响应信息及函数调用信息
def process_and_print_streaming_response(response):
    content = ""
    last_sender = ""

    for chunk in response:
        if "sender" in chunk:
            last_sender = chunk["sender"]

        if "content" in chunk and chunk["content"] is not None:
            if not content and last_sender:
                print(f"\033[94m{last_sender}:\033[0m", end=" ", flush=True)
                last_sender = ""
            print(chunk["content"], end="", flush=True)
            content += chunk["content"]

        if "tool_calls" in chunk and chunk["tool_calls"] is not None:
            for tool_call in chunk["tool_calls"]:
                f = tool_call["function"]
                name = f["name"]
                if not name:
                    continue
                print(f"\033[94m{last_sender}: \033[95m{name}\033[0m()")

        if "delim" in chunk and chunk["delim"] == "end" and content:
            print()  # End of response message
            content = ""

        if "response" in chunk:
            return chunk["response"]

# 格式化输出消息，用不同颜色显示智能体的回复信息和工具函数调用信息
def pretty_print_messages(messages) -> None:
    for message in messages:
        if message["role"] != "assistant":
            continue

        # print agent name in blue
        print(f"\033[94m{message['sender']}\033[0m:", end=" ")

        # print response, if any
        if message["content"]:
            print(message["content"])

        # print tool calls in purple, if any
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) > 1:
            print()
        for tool_call in tool_calls:
            f = tool_call["function"]
            name, args = f["name"], f["arguments"]
            arg_str = json.dumps(json.loads(args)).replace(":", "=")
            print(f"\033[95m{name}\033[0m({arg_str[1:-1]})")

# 开启用户输入循环，将输入发送给指定的智能体，处理其响应，并动态更新对话上下文。
def run_demo_loop(
    starting_agent, context_variables=None, stream=False, debug=False
) -> None:
    cclient = OpenAI(base_url='https://api-inference.modelscope.cn/v1/',
                     api_key='ms-3acb85d6-65f3-498d-88d2-fdfded95b9c4')
    # cclient = OpenAI(
    #     base_url="https://api.chatanywhere.tech/v1",
    #     api_key="sk-764egUqu5m6saadyCyiKn1xFewbcGD9CwVryxWkRBrhEZJl8",
    # )
    client = Swarm(client=cclient)
    #　client = Swarm()
    print("Starting Swarm CLI 🐝")

    messages = []
    agent = starting_agent

    while True:
        user_input = input("\033[90mUser\033[0m: ")
        messages.append({"role": "user", "content": user_input})

        response = client.run(
            agent=agent,
            messages=messages,
            context_variables=context_variables or {},
            stream=stream,
            debug=debug,
        )

        if stream:
            response = process_and_print_streaming_response(response)
        else:
            pretty_print_messages(response.messages)

        messages.extend(response.messages)
        agent = response.agent
        context_variables = response.context_variables
