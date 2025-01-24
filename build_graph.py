import json

def build_local_map_ke(data_path):
    data_file = f'{data_path}log_data.json'
    exer_n = 2791


    temp_list = []
    with open(data_file, encoding='utf8') as i_f:
        data = json.load(i_f)
    k_from_e = '' # e(src) to k(dst)
    e_from_k = '' # k(src) to k(dst)
    for line in data:
        for log in line['logs']:
            exer_id = log['exer_id'] - 1
            for k in log['knowledge_code']:
                if (str(exer_id) + '\t' + str(k + exer_n)) not in temp_list or (str(k + exer_n) + '\t' + str(exer_id)) not in temp_list:
                    k_from_e += str(exer_id) + '\t' + str(k + exer_n) + '\n'
                    e_from_k += str(k + exer_n) + '\t' + str(exer_id) + '\n'
                    temp_list.append((str(exer_id) + '\t' + str(k + exer_n)))
                    temp_list.append((str(k + exer_n) + '\t' + str(exer_id)))
    with open(f'{data_path}graph/k_from_e.txt', 'w') as f:
        f.write(k_from_e)
    with open(f'{data_path}graph/e_from_k.txt', 'w') as f:
        f.write(e_from_k)
def build_local_map_ue(data_path):
    data_file = f'{data_path}log_data.json'
    exer_n = 2791


    # e
    # u
    with open(data_file, encoding='utf8') as i_f:
        data_raw = json.load(i_f)
    data = []
    for stu in data_raw:
        records = stu['logs']
        user_id = stu['user_id']
        for log in records:
            data.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                            'knowledge_code': log['knowledge_code']})
    u_from_e = '' # e(src) to k(dst)
    e_from_u = '' # k(src) to k(dst)
    print (len(data))
    for line in data:
        # print(line)
        exer_id = line['exer_id'] - 1
        user_id = line['user_id'] - 1
        for k in line['knowledge_code']:
            u_from_e += str(exer_id) + '\t' + str(user_id + exer_n) + '\n'
            e_from_u += str(user_id + exer_n) + '\t' + str(exer_id) + '\n'
    with open(f'{data_path}graph/u_from_e.txt', 'w') as f:
        f.write(u_from_e)
    with open(f'{data_path}graph/e_from_u.txt', 'w') as f:
        f.write(e_from_u)


if __name__ == '__main__':
    data_path = 'data/junyi/'
    build_local_map_ke(data_path)
    build_local_map_ue(data_path)