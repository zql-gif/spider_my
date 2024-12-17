
"""
用智谱LLM对spider的test数据集作一个简单的baseline测试，不涉及提示工程等

"""
import json
import os
import random
from dataclasses import replace

from zhipuai import ZhipuAI
import tiktoken
from json_repair import repair_json
import logging
import re
import ast
import argparse

log = logging.getLogger(__name__)
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在目录
current_dir = os.path.dirname(current_file_path)

def try_parse_ast_to_json(function_string: str) -> tuple[str, dict]:
    """
     # 示例函数字符串
    function_string = "tool_call(first_int={'title': 'First Int', 'type': 'integer'}, second_int={'title': 'Second Int', 'type': 'integer'})"
    :return:
    """

    tree = ast.parse(str(function_string).strip())
    ast_info = ""
    json_result = {}
    # 查找函数调用节点并提取信息
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function_name = node.func.id
            args = {kw.arg: kw.value for kw in node.keywords}
            ast_info += f"Function Name: {function_name}\r\n"
            for arg, value in args.items():
                ast_info += f"Argument Name: {arg}\n"
                ast_info += f"Argument Value: {ast.dump(value)}\n"
                json_result[arg] = ast.literal_eval(value)

    return ast_info, json_result



def try_parse_json_object(input: str) -> tuple[str, dict]:
    """JSON cleaning and formatting utilities."""
    # Sometimes, the LLM returns a json string with some extra description, this function will clean it up.

    result = None
    try:
        # Try parse first
        result = json.loads(input)
    except json.JSONDecodeError:
        log.info("Warning: Error decoding faulty json, attempting repair")

    if result:
        return input, result

    _pattern = r"\{(.*)\}"
    _match = re.search(_pattern, input)
    input = "{" + _match.group(1) + "}" if _match else input

    # Clean up json string.
    input = (
        input.replace("{{", "{")
        .replace("}}", "}")
        .replace('"[{', "[{")
        .replace('}]"', "}]")
        .replace("\\", " ")
        .replace("\\n", " ")
        .replace("\n", " ")
        .replace("\r", "")
        .strip()
    )

    # Remove JSON Markdown Frame
    if input.startswith("```"):
        input = input[len("```"):]
    if input.startswith("```json"):
        input = input[len("```json"):]
    if input.endswith("```"):
        input = input[: len(input) - len("```")]

    try:
        result = json.loads(input)
    except json.JSONDecodeError:
        # Fixup potentially malformed json string using json_repair.
        json_info = str(repair_json(json_str=input, return_objects=False))

        # Generate JSON-string output using best-attempt prompting & parsing techniques.
        try:

            if len(json_info) < len(input):
                json_info, result = try_parse_ast_to_json(input)
            else:
                result = json.loads(json_info)

        except json.JSONDecodeError:
            log.exception("error loading json, json=%s", input)
            return json_info, {}
        else:
            if not isinstance(result, dict):
                log.exception("not expected dict type. type=%s:", type(result))
                return json_info, {}
            return json_info, result
    else:
        return input, result


def load_db_schema_info(gold_tables_file, db_id):
    with open(gold_tables_file, "r", encoding="utf-8") as r:
        contents = json.load(r)
    for content in contents:
        if content["db_id"].lower() == db_id.lower():
            return content
    return ""

def zhipu_text_to_sql_agent(client, model, temperature, content, gold_tables_file):
    # prompt 模板 ：包含question，基本的schema信息和解释
    db_schema = load_db_schema_info(gold_tables_file, content["db_id"])

    description = """
    database schema json contains the following information for each database:
    db_id: database id
    table_names_original: original table names stored in the database.
    table_names: cleaned and normalized table names. We make sure the table names are meaningful. [to be changed]
    column_names_original: original column names stored in the database. Each column looks like: [0, "id"]. 0 is the index of table names in table_names, which is city in this case. "id" is the column name.
    column_names: cleaned and normalized column names. We make sure the column names are meaningful. [to be changed]
    column_types: data type of each column
    foreign_keys: foreign keys in the database. [3, 8] means column indices in the column_names. These two columns are foreign keys of two different tables.
    primary_keys: primary keys in the database. Each number is the index of column_names.
    """

    llm_string = f"""
    Please generate the corresponding SQLite sql for the following question based on the provided database schema information and schema description, and provide a brief explanation in 150 words.\
    question : {content["question"]}\
    您应该始终遵循指令并输出一个有效的JSON对象,请默认使用 {{"sql": "$your_sql","explanation":"$your_explanation"}}。
    database schema description : {description}\
    database schema : {str(db_schema)}
    """

    # 获取格式化后的 prompt 字符串
    encoding = tiktoken.get_encoding('cl100k_base')
    # 计算格式化后的 prompt 的 token 数量
    tokens = encoding.encode(llm_string)
    # 如果 token 数量超过最大限制，则裁剪
    if len(tokens) > 4000:
        # 截断为最大 token 数
        truncated_tokens = tokens[:400]
        limited_llm_string = encoding.decode(truncated_tokens)
    else:
        limited_llm_string = llm_string

    response = client.chat.completions.create(
        model=model,  # 请填写您要调用的模型名称
        temperature=temperature,
        max_tokens=4095,
        messages=[
            {"role": "system", "content": "Let's think step by step.You are an expert in sqls."},
            {"role": "user", "content": limited_llm_string}
        ],
    )
    response_content = response.choices[0].message.content.replace("```", "").replace("json", "")
    _, response_json = try_parse_json_object(response_content)
    return response_json


def zhipu_text_to_sql_process(test_cnt, model, temperature, llm_key):
    gold_file = os.path.join(current_dir, "..", "spider_data", "test.json")
    gold_tables_file = os.path.join(current_dir, "..", "spider_data", "test_tables.json")
    output_dic = os.path.join(current_dir,"..","Output",model.lower())
    # 检测目录是否存在
    if not os.path.exists(output_dic):
        # 创建目录
        os.makedirs(output_dic, exist_ok=True)
    gold_file_output = os.path.join(output_dic, "gold.txt")
    predicted_file = os.path.join(output_dic, "predict.txt")
    response_file= os.path.join(output_dic, "predict.jsonl")

    with open(gold_file, "r", encoding="utf-8") as r:
        contents = json.load(r)
    if os.path.exists(predicted_file):
        with open(predicted_file, "r", encoding="utf-8") as r:
            lines = r.readlines()
        finished_cnt = len(lines)
    else:
        finished_cnt = 0

    chat = ZhipuAI(api_key=llm_key)  # 请填写您自己的APIKey

    while finished_cnt < test_cnt and finished_cnt < len(contents):
        print("text-to-sql task: " + str(finished_cnt))
        response = zhipu_text_to_sql_agent(chat, model, temperature, contents[finished_cnt], gold_tables_file)
        with open(gold_file_output, "a", encoding="utf-8") as w:
            w.write(contents[finished_cnt]["query"]+"\t"+contents[finished_cnt]["db_id"]+"\n")
        with open(predicted_file, "a", encoding="utf-8") as w:
            w.write(response["sql"]+"\n")
        with open(response_file, "a", encoding="utf-8") as w:
            json.dump(response, w)
            w.write("\n")
        finished_cnt += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run chatgpt_text_to_sql_process function.")

    # 添加命令行参数
    parser.add_argument('--test_num', dest='test_num', type=int, required=True)
    parser.add_argument('--model', dest='model', type=str, required=True)
    parser.add_argument('--temperature', dest='temperature', type=float, required=True)
    parser.add_argument('--llm_key', dest='llm_key', type=str, required=True)

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数并传递参数
    zhipu_text_to_sql_process(
        test_cnt=args.test_num,
        model=args.model,
        temperature=args.temperature,
        llm_key=args.llm_key
    )
