def extract_first_100000_lines(source_file, target_file, encoding='utf-8'):
    try:
        with open(source_file, 'r', encoding=encoding) as src:
            with open(target_file, 'w', encoding=encoding) as tgt:
                # 逐行读取并写入，直到达到2000行
                for line_num, line in enumerate(src, 1):
                    tgt.write(line)
                    if line_num >= 100000:
                        break
        print(f"成功提取前100000行到 {target_file}")
    except FileNotFoundError:
        print(f"错误：文件 {source_file} 不存在")
    except UnicodeDecodeError:
        print("错误：文件编码不匹配，请尝试指定正确的编码（如gbk）")

if __name__ == "__main__":
    extract_first_100000_lines("E:\download\HDFS_v1\HDFS.log","E:\swarm\swarm\logAgents\data\logs\HDFS_v1.log")