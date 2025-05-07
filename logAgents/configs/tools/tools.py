# version 1
#
# def read_log_file(file_url=None):
#     """读取日志文件的函数
#     Args:
#         file_url (str, optional): 日志文件路径. 默认为None.
#
#     Returns:
#         str/list: 返回文件内容，如果路径无效返回None
#     """
#     if file_url is None:
#         print("请提供日志文件路径")
#         return None
#
#     try:
#         with open(file_url, 'r', encoding='utf-8') as f:
#             # 返回行列表
#             return f.readlines()
#             # 或者返回整个文本内容
#             # return f.read()
#     except FileNotFoundError:
#         print(f"错误：文件 {file_url} 不存在")
#         return None
#     except Exception as e:
#         print(f"读取文件时发生错误: {str(e)}")
#         return None
#
#
# def find_format():
#     print("finding format……")
#     return "format识别完成"
#
#
# def log_parse():
#     print("parsing log……")
#     return "日志解析完成"
#
#
# def anomaly_detect():
#     print("log anomaly detecting……")
#     return "日志异常检测完成"
#
#
# def parse_resolve():
#     return "Parse resolved. No further questions."
#
#
# def detection_resolve():
#     return "Detection resolved. No further questions."


# version 2
import os
from swarm.types import Result

# def read_log(context_variables: dict,file_url=None):
#     """获取日志文件的函数
#     Args:
#         context_variables (dict): 上下文变量字典.
#         file_url (str, optional): 日志文件路径. 默认为None.
#     """
#     if file_url:
#         try:
#             if os.path.isfile(file_url) and os.access(file_url, os.R_OK):
#                 context_variables["log_file"] = file_url
#             else:
#                 print(f"Invalid file path or access denied: {file_url}")
#         except Exception as e:
#             print(f"Error validating file: {str(e)}")
#
#     print(context_variables)
#     return Result(context_variables=context_variables)


def read_log(file_url=None):
    """获取日志文件的函数
    Args:
        file_url (str, optional): 日志文件路径. 默认为None.
    """
    if file_url:
        try:
            if os.path.isfile(file_url) and os.access(file_url, os.R_OK):
                return Result(
                    value="Done",
                    context_variables={"log_file": file_url}
                )
            else:
                print(f"Invalid file path or access denied: {file_url}")
                # 返回错误信息并清除log_file
                return Result(
                    value="Invalid path",
                    context_variables={"log_file": None}
                )
        except Exception as e:
            print(f"Error validating file: {str(e)}")
            # 异常时清除log_file
            return Result(
                value="Error",
                context_variables={"log_file": None}
            )

    # 当file_url为None时，也清除log_file
    return Result(
        value="Invalid file path!",
        context_variables={"log_file": None}
    )


# def read_format_regex(context_variables: dict,format=None,regex=None):
#     """获取日志 format 和 regex 的函数
#     Args:
#         context_variables (dict): 上下文变量字典。
#         format (str, optional): 日志 format. 默认为None.
#         regex (list, optional): 用于过滤一些日志数据中无关信息的正则表达式，默认为 None.
#     """
#     if format is not None:
#         context_variables["log_format"] = format
#
#     if regex is not None:
#         context_variables["regex"] = regex
#
#     return Result(context_variables=context_variables)

def read_format_regex(format=None,regex=None):
    """获取日志 format 和 regex 的函数
    Args:
        format (str, optional): 日志 format. 默认为None.
        regex (list, optional): 用于过滤一些日志数据中无关信息的正则表达式，默认为 None.
    """
    if(format==None or regex==None):
        return Result(
            value="format或regex缺失"
        )

    return Result(
        value="done",
        context_variables={"log_format":format,"regex":regex}
    )


# def get_default_format_regex(context_variables: dict):
#     """获取默认日志 format 和 regex 的函数
#     Args:
#         context_variables (dict): 上下文变量字典。
#     """
#     context_variables["log_format"] = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
#     context_variables["regex"] = [
#         r'blk_(|-)[0-9]+' , # block id
#         r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
#         r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
#     ]
#
#     return context_variables


# def log_parse(context_variables: dict):
#     print("parsing log……\n")
#
#     input_dir = context_variables.get("log_file")  # The input directory of log file
#     output_dir = 'parse_result/'  # The output directory of parsing results
#     log_file = 'HDFS_2k.log'  # The input log file name
#     log_format = context_variables.get("log_format")  # HDFS log format
#     # Regular expression list for optional preprocessing (default: [])
#     regex = context_variables.get("regex")
#     st = 0.5  # Similarity threshold
#     depth = 4  # Depth of all leaf nodes
#
#     parser = LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
#     parser.parse(log_file)
#
#
#     print(context_variables)
#     return "日志解析完成"


# def anomaly_detect(context_variables: dict):
#     print("log anomaly detecting……\n")
#     print(context_variables)
#     return "日志异常检测完成"