import json
from tqdm import tqdm
# 从JSON文件中读取学生数据
with open('train_set.json', 'r') as file:
    students_data = json.load(file)
    
# result = {}
# for item in students_data:
#     user_id = item["user_id"]
#     exer_ids = [log["score"] for log in item["logs"]]
#     result[user_id] = exer_ids
# with open('encode_score.json', 'w') as file:
#     json.dump(result, file)
knowledge_code_result = {}
for item in students_data:
    user_id = item["user_id"]
    # Since each knowledge_code list is assumed to have only one element, we take the first one
    knowledge_codes = [log["knowledge_code"][0] for log in item["logs"]]
    knowledge_code_result[user_id] = knowledge_codes
with open('encode_kn.json', 'w') as file:
    json.dump(knowledge_code_result, file)