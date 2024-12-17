## links

Spider相关链接：
* github链接： [taoyds/spider: scripts and baselines for Spider: Yale complex and cross-domain semantic parsing and text-to-SQL challenge](https://github.com/taoyds/spider)
* 数据集网站： [Spider: Yale Semantic Parsing and Text-to-SQL Challenge](https://yale-lily.github.io//spider)
* spider的简单概括：[Text-to-SQL学习整理（八）Spider数据集介绍导语 前面的一系列博客中，我们已经了解到Text2SQL任务的 - 掘金](https://juejin.cn/post/7085557671528660999)
* spider的数据集大小说明：[Spider数据集论文研读 - 阿帆fann - 博客园](https://www.cnblogs.com/tyfann/p/15727093.html)

相关论文：
* Spider ：[[1809.08887] Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task](https://arxiv.org/abs/1809.08887)
* [Can LLM already serve as a database interface? a big bench for large-scale database grounded text-to-SQLs | Proceedings of the 37th International Conference on Neural Information Processing Systems](https://dl.acm.org/doi/10.5555/3666122.3667957)

Spider2.0原文：Notably, methods based on GPT-4 achieved execution accuracy of 91.2% and 73.0% on the classic benchmarks Spider 1.0 (Yu et al., 2018) and BIRD (Li et al., 2024b), respectively.

## 在Spider1.0数据集上简单测试 raw llm的text-to-sql能力
### Spider1.0
关于Spider1.0数据集的介绍，参考 [spider/README.md at master · taoyds/spider](https://github.com/taoyds/spider/blob/master/README.md)
本项目下的文件夹 `spider_data` 下载于链接 [spider_data.zip - Google 云端硬盘](https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view)

### preparation
Spider1.0的数据集spider_data包含的数据库文件比较大，不便上传到github，所以请先在下面链接下载spider_data.zip文件并解压到本项目目录下： [spider_data.zip - Google 云端硬盘](https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view)
### setup
1. 测试数据取自`spider_data`中的测试文件test.json ,  对应的database schema文件test_tables.json以及test_gold.sql
2. input : Spider1.0数据集的测试文件test.json ,  以及对应的database schema文件test_tables.json
3. output : 生成的sql文件 "test_predict_1.0.txt" ，llm的生成结果和代价的记录文件test_predict_1.0.jsonl
4. llm setup 
* model = "gpt-4o-mini"， “gpt-4-turbo” , 智谱的"glm-4-plus"
* temperature = 0.0
* 测试数目：150条
### prompt(llm代码见文件夹Spider1_LLM_Baseline)
只包括简单的task description， question，database schema（直接采用spider数据集提供的json格式schema） and description
``` python
llm_string = """  
Let's think step by step.You are an expert in sqls.\  
Please generate the corresponding SQLite sql for the following question based on the provided database schema information and schema description, and provide a brief explanation.\  
question : {question}\  
database schema : {schema}\  
database schema description : {description}  
Answer the following information: {format_instructions}  """  
  
prompt_template = ChatPromptTemplate.from_template(llm_string)  
  
response_schemas = [  
    ResponseSchema(type="string", name="sql", description='The sql answer to the question.'),  
    ResponseSchema(type="string", name="explanation", description='Explain the basis for the sql answer.')  
]
```

### run llm agent
下面分别是基于gpt-4o-mini, gpt-4,  glm-4-plus（智谱）为Spider测试sql执行text-to-sql的指令
``` shell
cd <project_directory>
```

``` shell
python -m Spider1_LLM_Baseline.chatgpt_text_to_sql --test_num 150 --model "gpt-4o-mini" --temperature 0.0 --"llm_key" ${your_api_key}
```


``` shell
python -m Spider1_LLM_Baseline.chatgpt_text_to_sql --test_num 150 --model "gpt-4-turbo" --temperature 0.0 --"llm_key" ${your_api_key}
```


``` shell
python -m Spider1_LLM_Baseline.zhipu_text_to_sql --test_num 150 --model "glm-4-plus" --temperature 0.0 --"llm_key" ${your_api_key}
```

上述指令参数说明：

| option          | description                  |
| --------------- | ---------------------------- |
| `--test_num`    | 从spider_data的测试数据集中选取的测试数据条数 |
| `--temperature` | Temperature for LLM          |
| `--model`       | Model to use for LLM         |
| `--llm_key`     | llm key                      |


运行结果将输出到文件夹`Output`的`model_name`子文件夹中，如下：
gold.txt : 指定测试个数的sqls正确答案
predixt.txt : 指定测试个数的sqls预测答案
predixt.jsonl ：指定测试个数的text-to-sql任务中间结果

### evaluate
上面三个测试模型的evaluate指令分别如下：
``` shell
cd <project_directory>
```

``` shell
python evaluation.py --gold "Output/gpt-4o-mini/gold.txt" --pred "Output/gpt-4o-mini/predict.txt" --acc "Output/gpt-4o-mini/eval_result.txt" --db "spider_data/test_database" --etype "all" --table "spider_data/test_tables.json"
```

``` shell
python evaluation.py --gold "Output/gpt-4-turbo/gold.txt" --pred "Output/gpt-4-turbo/predict.txt" --acc "Output/gpt-4/eval_result.txt" --db "spider_data/test_database" --etype "all" --table "spider_data/test_tables.json"
```

``` shell
python evaluation.py --gold "Output/gpt-4o-mini/gold.txt" --pred "Output/glm-4-plus/predict.txt" --acc "Output/glm-4-plus/eval_result.txt" --db "spider_data/test_database" --etype "all" --table "spider_data/test_tables.json"
```

上述指令的输出结果存储于文件夹`Output`的`model_name`子文件夹中，如下：
eval_result.txt : 指定测试个数的sqls的text-to-sql任务评估结果

### results
"gpt-4o-mini"
```
                     easy                 medium               hard                 extra                all                   
count                36                   53                   35                   26                   150                   
=====================   EXECUTION ACCURACY     =====================  
execution            0.444                0.151                0.086                0.000                0.180                 
  
====================== EXACT MATCHING ACCURACY =====================  
exact match          0.444                0.151                0.086                0.000                0.180
```

"gpt-4-turbo"
```
                     easy                 medium               hard                 extra                all                   
count                28                   34                   29                   19                   110                   
=====================   EXECUTION ACCURACY     =====================  
execution            0.607                0.206                0.207                0.105                0.291                 
  
====================== EXACT MATCHING ACCURACY =====================  
exact match          0.607                0.235                0.207                0.105                0.300
```

智谱的"glm-4-plus"
```
                     easy                 medium               hard                 extra                all                   
count                36                   53                   35                   26                   150                   
=====================   EXECUTION ACCURACY     =====================  
execution            0.528                0.226                0.000                0.269                0.253                 
  
====================== EXACT MATCHING ACCURACY =====================  
exact match          0.444                0.245                0.000                0.231                0.233
```


* 直接使用上面这种prompt，database schema信息较长，很容易超过问题的上下文限制（8000+，4000+）
* 各难度测试样例的平均execution success 比例，最佳情况下也只有30%左右