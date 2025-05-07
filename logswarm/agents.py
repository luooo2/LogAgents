from swarm import Agent


def transfer_to_loglizer_agent():
    """When the user needs to detect log anomalies, this function is called after the log is parsed to log templates and structured logs."""
    return loglizer_agent

def parse_log():
    """when the user needs to parse log, this function is called to parse log into log templates and structured logs."""
    print("parsing log……")

def transfer_to_parser_agent():
    return parser_agent

interact_agent = Agent(
    name="Interact Agent",
    model="gpt-4o-mini",
    instructions="You are a customer service, responsible for submitting the input logs to parser_agent for parsing.",
    functions=[transfer_to_parser_agent],
)

parser_agent = Agent(
    name="Parser Agent",
    model="gpt-4o-mini",
    instructions="Log parsing refers to the process of converting semi-structured log messages into structured log events. You are very good at log parsing. When users want to parse logs, you can output parsing results (log templates, structured logs) based on the input log files; When users want to perform log anomaly detection, you can output parsing results (log templates, structured logs) based on the input log files, and provide the parsing results to the loglizer_agent for anomaly detection.",
    functions=[transfer_to_loglizer_agent,parse_log],
)

loglizer_agent = Agent(
    name="Loglizer",
    model="gpt-4o-mini",
    instructions="Log anomaly detection refers to the process of analyzing log data to identify potential abnormal behaviors or events within it. You are very good at log anomaly detection. When users want to perform log anomaly detection, you can perform log anomaly detection and analyze which parts of the logs may be abnormal based on the parsed results (log templates, structured logs)."
)

