from swarm import Agent
from logAgents.data.prompts.prompts import *
from logAgents.configs.tools.tools import *
from logAgents.configs.tools.parse.drain import log_parse
from logAgents.configs.tools.detect.detect import anomaly_detect

# version1：仿airline workflow， 与用户交互
#
# def transfer_to_log_anomaly_detection():
#     """当用户希望进行日志异常检测时，调用该函数。"""
#     return loglizer_agent
#
#
# def transfer_to_log_parser():
#     """当需要进行日志解析时，调用该函数。"""
#     return parser_agent
#
#
# def transfer_to_triage():
#     """当用户需要别的服务时，调用该函数。例如，如果用户请求的业务不是当前agent处理的时，调用该函数。"""
#     return triage_agent
#
#
# triage_agent = Agent(
#     name="Triage Agent",
#     model="Qwen/Qwen2.5-32B-Instruct",
#     instructions="""
#     你需要对用户请求进行分类，并调用一个工具来转移到正确的agent。
#     一旦你准备好转移到正确的agent，调用工具转移到正确的agent。
#     你不需要知道具体细节，只需要知道请求的主题。
#     当你需要更多信息来对请求进行分类时，直接问一个问题，而不解释你为什么要问。
#     不要与用户分享你的思考过程！不要代表用户做出不合理的假设。
#     """,
#     functions=[transfer_to_log_anomaly_detection,transfer_to_log_parser]
# )
#
#
# parser_agent = Agent(
#     name="Parser Agent",
#     model="Qwen/Qwen2.5-32B-Instruct",
#     instructions=PARSER_SYSTEM_PROMPTS,
#     functions=[transfer_to_triage,find_format,log_parse,parse_resolve],
# )
#
#
# loglizer_agent = Agent(
#     name="Loglizer",
#     model="Qwen/Qwen2.5-32B-Instruct",
#     instructions=ANOMALY_DETECTION_SYSTEM_PROMPTS,
#     functions=[transfer_to_triage,detection_resolve,anomaly_detect],
# )



# version2：顺序执行workflow

def transfer_to_log_anomaly_detection():
    """当进入异常检测步骤时，调用该函数。"""
    return loglizer_agent


def transfer_to_log_parser():
    """当进入日志解析步骤时，调用该函数。"""
    return parser_agent


def transfer_to_customer():
    """当用户想要与客服进行交流或服务完成时，调用该函数。"""
    return customer_agent


customer_agent = Agent(
    name="Customer Agent",
    model="Qwen/Qwen2.5-32B-Instruct",
    instructions=PUBLIC_SYSTEM_PROMPTS+CUSTOMER_SYSTEM_PROMPTS,
    functions=[transfer_to_log_parser]
)

parser_agent = Agent(
    name="Parser Agent",
    model="Qwen/Qwen2.5-32B-Instruct",
    instructions=PUBLIC_SYSTEM_PROMPTS+PARSER_SYSTEM_PROMPTS,
    functions=[transfer_to_log_anomaly_detection,read_log,read_format_regex,log_parse],
)

loglizer_agent = Agent(
    name="Loglizer",
    model="Qwen/Qwen2.5-32B-Instruct",
    instructions=PUBLIC_SYSTEM_PROMPTS+ANOMALY_DETECTION_SYSTEM_PROMPTS,
    functions=[transfer_to_customer,anomaly_detect],
)
















