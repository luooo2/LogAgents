from openai import OpenAI
from swarm import Swarm, Agent
from agents import interact_agent,parser_agent,loglizer_agent
import sys
import os
import json

def pretty_print_messages(messages):
    for message in messages:
        if message["content"] is None:
            continue
        print(f"{message['sender']}: {message['content']}")

def main():
    if len(sys.argv)<5:
        print("Usage: python anomaly_extraction.py <input_message> --log_path <log_path> [--output <output_file_path>]")
        sys.exit(1)

    # 获取命令行参数
    input_message = sys.argv[1]
    log_path = sys.argv[sys.argv.index("--log_path") + 1]
    # log_format = sys.argv[sys.argv.index("--log_format") + 1]
    # log_regex = sys.argv[sys.argv.index("--log_regex") + 1]

    # 检查是否提供了输出文件路径
    output_file_path = None
    if "--output" in sys.argv:
        try:
            output_file_path = sys.argv[sys.argv.index("--output") + 1]
        except IndexError:
            print("Error: No output file path provided after '--output'.")
            sys.exit(1)

    # 检查文件路径是否有效
    if not os.path.exists(log_path):
        print(f"Error: File '{log_path}' doesn't exist.")
        sys.exit(1)

    # 读取文件内容
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            input_file_content = f.read()
    except Exception as e:
        print(f"Error reading file {log_path}: {e}")
        sys.exit(1)

    cclient = OpenAI(
        base_url="https://api.chatanywhere.tech/v1",
        api_key="sk-764egUqu5m6saadyCyiKn1xFewbcGD9CwVryxWkRBrhEZJl8",
    )
    client = Swarm(client=cclient)

    # 构建对话历史，将任务描述和文件内容作为用户输入
    chat_histroy = [
        {"role": "user", "content": input_message},
        {"role": "user", "content": f"Here is the log file content:\n{input_file_content}\n"}
    ]
    context_variables = {}

    agent = interact_agent

    # 调用 Swarm 进行对话
    print("Processing your request...")
    response = client.run(agent=agent,
                          messages=chat_histroy,
                          context_variables=context_variables,
                          debug=True,)

    # 收集模型的回复内容
    assistant_responses = []
    for msg in response.messages:
        if msg["role"] == "assistant":
            # print(f"\nAssistant Response:\n{msg['content']}\n")
            assistant_responses.append(msg["content"])

    # 合并所有回复
    final_result = "\n".join([response for response in assistant_responses if response is not None])

    # 输出模型回复
    print("\nAssistant Response:\n")
    print(final_result)

    # 保存结果到文件（如果指定了 --output 参数）
    if output_file_path:
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(final_result)
            print(f"\nResult saved to file: {output_file_path}")
        except Exception as e:
            print(f"Error saving result to file {output_file_path}: {e}")
            sys.exit(1)


if __name__=="__main__":
    main()


    #
    #
    # messages = []
    # agent = parser_agent
    # while True:
    #     user_input = input("> ")
    #     messages.append({"role": "user", "content": user_input})
    #     response = client.run(agent=agent, messages=messages)
    #     messages = response.messages
    #     agent = response.agent
    #     pretty_print_messages(messages)
    #
    #
