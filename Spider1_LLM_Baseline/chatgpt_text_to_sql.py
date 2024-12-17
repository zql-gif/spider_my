
"""
用chatgpt对spider的test数据集作一个简单的baseline测试，不涉及提示工程等
input :
   test.json,存有测试数据的详细信息，包括"db_id"，"query"，"question"等等
           {
            "db_id": "soccer_3",
            "query": "SELECT count(*) FROM club",
            "query_toks": [
            ],
            "query_toks_no_value": [
            ],
            "question": "How many clubs are there?",
            "question_toks": [
            ],
            "sql": {
            }
        }

   test_tables.json, 存有测试集所需的database schema信息
output :
   test_predict_1.0.sql: 存储对input内的数据进行text-to-sql任务转换的结果
"""
import json
import os
import random
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
import openai
import tiktoken
from openai import OpenAI
import argparse


# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在目录
current_dir = os.path.dirname(current_file_path)

def load_db_schema_info(gold_tables_file, db_id):
    with open(gold_tables_file, "r", encoding="utf-8") as r:
        contents = json.load(r)
    for content in contents:
        if content["db_id"].lower() == db_id.lower():
            return content
    return ""

def chatgpt_text_to_sql_agent(conversation, model, llm_key, content, gold_tables_file):
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

    llm_string = """
    Let's think step by step.You are an expert in sqls.\
    Please generate the corresponding SQLite sql for the following question based on the provided database schema information and schema description, and provide a brief explanation.\
    question : {question}\
    Answer the following information: {format_instructions}\
    database schema description : {description}\
    database schema : {schema}
    """

    prompt_template = ChatPromptTemplate.from_template(llm_string)

    response_schemas = [
        ResponseSchema(type="string", name="sql", description='The sql answer to the question.'),
        ResponseSchema(type="string", name="explanation", description='Explain the basis for the sql answer in less than 100 words.')
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt_messages = prompt_template.format_messages(
        question=content["question"],
        schema=str(db_schema),
        description=description,
        format_instructions=format_instructions
    )

    # 获取格式化后的 prompt 字符串
    encoding = tiktoken.get_encoding('cl100k_base')
    formatted_prompt = prompt_messages[0].content
    # 计算格式化后的 prompt 的 token 数量
    tokens = encoding.encode(formatted_prompt)
    # 如果 token 数量超过最大限制，则裁剪
    if len(tokens) > 7800:
        # 截断为最大 token 数
        truncated_tokens = tokens[:7800]
        formatted_prompt = encoding.decode(truncated_tokens)
    # 生成最终的 prompt 消息
    prompt_messages[0].content = formatted_prompt
    response = conversation.predict(input=prompt_messages[0].content)
    output_dict = output_parser.parse(response)
    return output_dict



def chatgpt_text_to_sql_process(test_cnt, model, temperature, llm_key):
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

    os.environ["OPENAI_API_KEY"] = llm_key
    chat = ChatOpenAI(temperature=temperature, model=model)
    conversation = ConversationChain(
        llm=chat,
        verbose=False  # 为true的时候是展示langchain实际在做什么
    )

    while finished_cnt < test_cnt and finished_cnt < len(contents):
        print("text-to-sql task: " + str(finished_cnt))
        cost = {}
        with get_openai_callback() as cb:
            response = chatgpt_text_to_sql_agent(conversation, model, llm_key, contents[finished_cnt], gold_tables_file)
            cost["Total Tokens"] = cb.total_tokens
            cost["Prompt Tokens"] = cb.prompt_tokens
            cost["Completion Tokens"] = cb.completion_tokens
            cost["Total Cost (USD)"] = cb.total_cost
            response["cost"] = cost
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
    chatgpt_text_to_sql_process(
        test_cnt=args.test_num,
        model=args.model,
        temperature=args.temperature,
        llm_key=args.llm_key
    )
