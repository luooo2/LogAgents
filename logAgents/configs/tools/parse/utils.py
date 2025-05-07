import os

def split_file_path(absolute_path):
    # 规范化路径，处理多余的斜杠或不同操作系统的分隔符
    normalized_path = os.path.normpath(absolute_path)
    # 分割路径和文件名
    input_dir, log_file = os.path.split(normalized_path)
    return input_dir, log_file