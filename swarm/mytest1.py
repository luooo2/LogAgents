# from openai import OpenAI
# client = OpenAI()
#
# completion = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "Say this is a test!"},
#   ]
# )
#
# print(completion.choices[0].message)
import os

from openai import OpenAI
from swarm import Agent, Swarm
from swarm.types import Result

cclient = OpenAI(base_url='https://api-inference.modelscope.cn/v1/',
                api_key='0336b506-55fd-4f1b-804b-436b8fea98a1')

client = Swarm(client=cclient)

def read_log(file_url=None):
  """获取日志文件的函数"""
  if file_url:
    try:
      if os.path.isfile(file_url) and os.access(file_url, os.R_OK):
        return Result(
          value="Done",
          context_variables={"log_file": file_url}  # 成功时更新
        )
      else:
        print(f"无效文件路径或权限不足: {file_url}")
        # 错误时不修改log_file（不返回该键）
        return Result(value="Invalid path")
    except Exception as e:
      print(f"文件验证错误: {str(e)}")
      # 异常时不修改log_file
      return Result(value="Error")

  # 未提供路径时清除log_file（根据需求决定）
  return Result(
    value="无效路径!",
    context_variables={"log_file": None}  # 明确清除
  )

agent = Agent(name="read_agent",functions=[read_log],model="Qwen/Qwen2.5-32B-Instruct")

response = client.run(
   agent=agent,
   messages=[{"role": "user", "content": "读取日志文件 E:\download\logparser-main\logparser-main\data\loghub_2k\HDFS\HDFS_2k.log"}],
   context_variables={"user_name": "John"}
)
print(response.agent.name)
print(response.context_variables)