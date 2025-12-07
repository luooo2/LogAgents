from logAgents.configs.tools.parse.utils import split_file_path
from logAgents.configs.tools.parse.Drain.Drain import LogParser
from swarm.types import Result
import os

def log_parse(context_variables: dict):
    print("parsing log……\n")

    file_log = context_variables.get("log_file")
    print(context_variables)
    input_dir, log_file = split_file_path(file_log) # The input directory of log file & The input log file name
    output_dir = 'parse_result/'  # The output directory of parsing results
    log_format = context_variables.get("log_format")  # log format
    # Regular expression list for optional preprocessing (default: [])
    regex = context_variables.get("regex")
    st = 0.5  # Similarity threshold
    depth = 4  # Depth of all leaf nodes

    parser = LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
    parser.parse(log_file)

    structured_log = os.path.join(output_dir, log_file + "_structured.csv")
    log_templates = os.path.join(output_dir, log_file + "_templates.csv")

    print(context_variables)
    return Result(
        value="done",
        context_variables={"structured_log":structured_log, "log_templates":log_templates}
    )