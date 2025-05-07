from configs.agents import *
from swarm.repl import run_demo_loop

context_variables = {
    "log_file":None,
    "log_format":"""<Date> <Time> <Pid> <Level> <Component>: <Content>""",
    "regex":[
        r'blk_(|-)[0-9]+' , # block id
        r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
        r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
    ],
    "structured_log":None,
    "log_templates":None,
}

if __name__ == "__main__":
    run_demo_loop(customer_agent,context_variables=context_variables, debug=True)
