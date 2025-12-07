import json
import os
import re
from typing import Dict, List, Any
from collections import defaultdict
from openai import OpenAI

os.environ["DASHSCOPE_API_KEY"] = "sk-0ba6f36eaf4449e283ed1dd565ac7c59"

class RootCauseAnalyzer:
    def __init__(self, model="qwen-turbo"):
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

#     def _build_cot_prompt_template(self,block_data) -> str:
#         """构建CoT思维链提示模板"""
#         return f"""
# 您是一个资深的分布式系统异常分析专家，请使用Chain-of-Thought思维链方法对以下HDFS日志异常进行根因分析。
# 请严格按照以下步骤进行推理，并在每个步骤后添加<thinking>标签说明推理过程：
#
# 输入数据格式：
# {{
#     "BlockId": "块ID",
#     "TotalLogs": 日志数量,
#     "FirstTimestamp": 起始时间,
#     "LastTimestamp": 结束时间,
#     "EventSequence": [事件序列],
#     "Logs": [
#         {{
#             "Time": 时间戳,
#             "EventId": "事件ID",
#             "Content": "日志内容",
#             "EventTemplate": "事件模板",
#             "Pid": 进程ID,
#             "Level": "日志级别"
#         }},
#         ...
#     ]
# }}
#
# 分析步骤：
# 1. **关键事件提取**：识别日志中的关键事件，特别是错误、警告和异常事件
# 2. **时间线分析**：分析事件发生的时序关系，寻找异常延迟或时间间隔
# 3. **事件模式识别**：发现重复出现的事件序列模式
# 4. **错误类型分类**：将错误信息归类（如网络、存储、资源等）
# 5. **依赖关系分析**：识别系统组件之间的依赖关系和影响链
# 6. **根因假设生成**：基于以上分析提出可能的根因假设
# 7. **证据验证**：评估每个假设与日志证据的匹配程度
# 8. **结论与建议**：确定最可能的根因并提供优化建议
#
# 请对以下异常块进行深入分析：
# {block_data}
#
# 分析报告格式：
# ### 关键事件提取
# <thinking>...</thinking>
# <events>
# - 事件ID: 描述 (关键程度)
# ...
# </events>
#
# ### 时间线分析
# <thinking>...</thinking>
# <timeline>
# - 重要时间点: 分析
# ...
# </timeline>
#
# ### 事件模式识别
# <thinking>...</thinking>
# <patterns>
# - 模式名称: 描述 (出现次数)
# ...
# </patterns>
#
# ### 错误类型分类
# <thinking>...</thinking>
# <errors>
# - 错误类别: 具体错误 (出现次数)
# ...
# </errors>
#
# ### 依赖关系分析
# <thinking>...</thinking>
# <dependencies>
# - 组件A → 组件B: 依赖关系描述
# ...
# </dependencies>
#
# ### 根因假设
# <thinking>...</thinking>
# <hypotheses>
# 1. 假设1: ... (支持证据)
# 2. 假设2: ... (支持证据)
# ...
# </hypotheses>
#
# ### 证据验证
# <thinking>...</thinking>
# <validation>
# - 假设1: 验证结果 (匹配度)
# ...
# </validation>
#
# ### 结论与建议
# <thinking>...</thinking>
# <conclusion>
# 根本原因: ...
# </conclusion>
# <recommendations>
# 1. 建议1: ...
# 2. 建议2: ...
# ...
# </recommendations>
# """
    def _build_block_cot_prompt(self, block_data: dict) -> str:
        """Construct a CoT prompt for block-level root cause analysis, return structured JSON output."""
        block_json = json.dumps(block_data, ensure_ascii=False, indent=2)
        return f'''
您是一位资深的分布式系统专家，请使用链式思考（Chain-of-Thought）方法对以下 HDFS 日志块进行根因分析。

日志块数据（JSON）：
{block_json}

请按以下步骤推理，并在每步后添加 <thinking> 标签：
1. 关键事件提取：识别错误、警告和异常事件。
2. 时间线分析：分析事件发生的时序关系和延迟。
3. 模式识别：发现重复出现的事件序列。
4. 错误分类：按网络、存储、资源等类型进行归类。
5. 依赖关系分析：描述组件之间的依赖及影响链。
6. 根因假设：提出可能的根因假设。
7. 证据验证：评估每个假设与日志的匹配度。
8. 结论与建议：确定最可能的根因并给出修复建议。

请严格输出以下结构化 JSON：
```
{{
    "block_id": string,
    "timestamp": string,
    "root_cause": string,
    "exception_type": string,
    "analysis_steps": [
        {{"step": int, "description": string, "thinking": string}},
        ...
    ],
    "injection_hint": {{
        "component": string,
    "method": string,
    "fault_type": string,
    "inject_time": string
    }}
}}
```'''

    def analyze_block(self, block_data: Dict) -> Dict:
        """分析单个日志块，并返回结构化结果。"""
        prompt = self._build_block_cot_prompt(block_data)
        print("-----prompt-----")
        print(prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                ],
                # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
                # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
                # extra_body={"enable_thinking": False},
            )
            print(response.model_dump_json())

            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            return {"error": str(e)}

    # def analyze_block(self, block_data: Dict) -> Dict:
    #     """分析单个异常块"""
    #     prompt = self._build_block_cot_prompt(block_data)
    #     print("-----prompt-----")
    #     print(prompt)
    #
    #     try:
    #         response = self.client.chat.completions.create(
    #             model=self.model,
    #             messages=[
    #                 {"role": "system", "content": prompt},
    #             ],
    #             # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
    #             # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
    #             # extra_body={"enable_thinking": False},
    #         )
    #         print(response.model_dump_json())
    #         # response = openai.ChatCompletion.create(
    #         #     model=self.model,
    #         #     messages=[{"role": "system", "content": prompt}],
    #         #     temperature=self.temperature,
    #         #     max_tokens=2000
    #         # )
    #         analysis = response.choices[0].message.content
    #         return self._parse_analysis(analysis)
    #     except Exception as e:
    #         return {"error": str(e)}

    # def _parse_analysis(self, analysis_text: str) -> Dict:
    #     """解析分析结果"""
    #     sections = [
    #         "关键事件提取", "时间线分析", "事件模式识别", "错误类型分类",
    #         "依赖关系分析", "根因假设", "证据验证", "结论与建议"
    #     ]
    #
    #     result = {}
    #     current_section = None
    #     content_buffer = []
    #
    #     lines = analysis_text.split("\n")
    #     for line in lines:
    #         # 检测新章节
    #         section_detected = False
    #         for section in sections:
    #             if line.startswith(f"### {section}"):
    #                 if current_section:
    #                     result[current_section] = "\n".join(content_buffer)
    #                 current_section = section
    #                 content_buffer = []
    #                 section_detected = True
    #                 break
    #
    #         if not section_detected and current_section:
    #             content_buffer.append(line)
    #
    #     if current_section and content_buffer:
    #         result[current_section] = "\n".join(content_buffer)
    #
    #     return result

#     def generate_system_report(self, blocks: List[Dict]) -> Dict:
#         """生成系统级根因分析报告"""
#         # 聚合所有块的初步分析
#         all_errors = defaultdict(int)
#         all_patterns = defaultdict(int)
#         critical_events = defaultdict(int)
#
#         for block in blocks:
#             analysis = self.analyze_block(block)
#
#             # 提取错误信息
#             if "错误类型分类" in analysis:
#                 errors_text = analysis["错误类型分类"]
#                 errors = re.findall(r"- (\w+): (.+?) \((\d+)次\)", errors_text)
#                 for category, error, count in errors:
#                     key = f"{category}:{error}"
#                     all_errors[key] += int(count)
#
#             # 提取事件模式
#             if "事件模式识别" in analysis:
#                 patterns_text = analysis["事件模式识别"]
#                 patterns = re.findall(r"- (.+?): (.+?) \((\d+)次\)", patterns_text)
#                 for pattern, desc, count in patterns:
#                     all_patterns[pattern] += int(count)
#
#             # 提取关键事件
#             if "关键事件提取" in analysis:
#                 events_text = analysis["关键事件提取"]
#                 events = re.findall(r"- (E\d+): (.+?) \(关键程度: (.+?)\)", events_text)
#                 for event_id, desc, criticality in events:
#                     critical_events[event_id] = max(
#                         critical_events.get(event_id, 0),
#                         self._criticality_to_score(criticality)
#                     )
#
#         # 准备系统级分析提示
#         system_prompt = f"""
# 您是一个资深的分布式系统架构师，请基于以下聚合分析结果为整个HDFS集群生成系统级根因报告：
#
# ### 错误统计
# {json.dumps(dict(all_errors), indent=2)}
#
# ### 事件模式统计
# {json.dumps(dict(all_patterns), indent=2)}
#
# ### 关键事件
# {json.dumps(dict(critical_events), indent=2)}
#
# 请按照以下步骤分析：
# 1. **热点分析**：识别最高频的错误类型和事件模式
# 2. **系统瓶颈**：确定可能的基础设施瓶颈（网络、存储、计算）
# 3. **关联分析**：发现错误之间的关联关系
# 4. **根因推断**：推断系统级根本原因
# 5. **优化建议**：提供架构级优化建议
#
# 报告格式：
# ### 系统级根因分析
# <analysis>...</analysis>
#
# ### 主要发现
# <findings>
# 1. ...
# </findings>
#
# ### 根本原因
# <root_cause>
# ...
# </root_cause>
#
# ### 架构优化建议
# <recommendations>
# 1. ...
# </recommendations>
# """
#
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                 ],
#                 # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
#                 # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
#                 # extra_body={"enable_thinking": False},
#             )
#             # print(response.model_dump_json())
#             # response = self.client.ChatCompletion.create(
#             #     model=self.model,
#             #     messages=[{"role": "system", "content": system_prompt}],
#             #     temperature=self.temperature,
#             #     max_tokens=1500
#             # )
#             report = response.choices[0].message.content
#             return self._parse_system_report(report)
#         except Exception as e:
#             return {"error": str(e)}

    # def _criticality_to_score(self, criticality: str) -> int:
    #     """将关键程度转换为分数"""
    #     mapping = {
    #         "低": 1,
    #         "中": 3,
    #         "高": 5,
    #         "严重": 7,
    #         "致命": 10
    #     }
    #     return mapping.get(criticality, 1)
    #
    # def _parse_system_report(self, report_text: str) -> Dict:
    #     """解析系统级报告"""
    #     sections = ["系统级根因分析", "主要发现", "根本原因", "架构优化建议"]
    #     result = {}
    #     current_section = None
    #     content_buffer = []
    #
    #     lines = report_text.split("\n")
    #     for line in lines:
    #         # 检测新章节
    #         section_detected = False
    #         for section in sections:
    #             if line.startswith(f"### {section}"):
    #                 if current_section:
    #                     result[current_section] = "\n".join(content_buffer)
    #                 current_section = section
    #                 content_buffer = []
    #                 section_detected = True
    #                 break
    #
    #         if not section_detected and current_section:
    #             content_buffer.append(line)
    #
    #     if current_section and content_buffer:
    #         result[current_section] = "\n".join(content_buffer)
    #
    #     return result

    def _build_system_prompt(self, block_results: list) -> str:
        """构建系统级异常分析提示模板，聚合块级结果并输出因果链和注入计划。"""
        blocks_json = json.dumps(block_results, ensure_ascii=False, indent=2)
        return f'''
您是一位系统级异常分析专家。已获得以下块级分析结果（JSON 数组）：
{blocks_json}

请使用链式思考完成以下步骤：
1. 按时间排序并聚合所有事件。
2. 识别跨块的因果链。
3. 确定系统级别的主要根因。
4. 提出可用于复现该故障的注入计划。

请严格输出以下结构化 JSON：
```
{{
    "summary_time_range": {{"start": string, "end": string}},
    "exception_chain": [
        {{"component": string, "event": string, "time": string}},
        ...
    ],
    "main_root_cause": string,
    "injection_plan": {{
        "fault_type": string,
        "component": string,
        "method": string,
        "inject_time": string
    }}
}}
```'''

    def generate_system_report(self, block_results: list) -> dict:
        """生成系统级根因分析报告。"""
        prompt = self._build_system_prompt(block_results)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
        )
        return json.loads(response.choices[0].message.content)


# 使用示例
if __name__ == "__main__":
    # 加载之前生成的异常块数据
    with open("detect_result/sample.json") as f:
        anomaly_blocks = json.load(f)

    # 初始化分析智能体
    analyzer = RootCauseAnalyzer()

    # 分析单个异常块（示例）
    sample_block = anomaly_blocks[0]
    block_analysis = analyzer.analyze_block(sample_block)

    print("单个异常块分析结果:")
    for section, content in block_analysis.items():
        print(f"\n===== {section} =====")
        print(content)

    # 生成系统级报告
    # 1. 批量块级分析
    block_results = []
    for block in anomaly_blocks:
        result = analyzer.analyze_block(sample_block)
        block_results.append(result)

    # 2. 将块级分析结果写入 JSON 文件
    with open("detect_result/block_results.json", "w", encoding="utf-8") as f:
        json.dump(block_results, f, ensure_ascii=False, indent=2)
    print("✅ 已保存所有块级分析结果到 block_results.json")

    # 3. 从文件中加载块级结果
    with open("detect_result/block_results.json", "r", encoding="utf-8") as f:
        loaded_results = json.load(f)

    # 4. 调用系统级分析 Agent
    system_report = analyzer.generate_system_report(loaded_results)
    print("\n\n===== 系统级根因分析报告 =====")
    print(system_report)

    # 5. 将系统级报告也保存或打印
    with open("detect_result/system_report.json", "w", encoding="utf-8") as f:
        json.dump(system_report, f, ensure_ascii=False, indent=2)
    print("✅ 已生成并保存系统级根因报告到 system_report.json")

