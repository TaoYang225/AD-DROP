import pandas as pd
import os
import numpy as np

def get_file_name(path): # 获取目录下的所有文件
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list

path_name = 'log_bert-base-uncased_GA_42'
file_list = sorted(get_file_name(path_name))

max_result = 0
max_name = ''
results = np.zeros((81,))
for i,file in enumerate(file_list):
    # print(file)
    with open(file_list[i], 'r') as f:
        data = f.read()
        best = float(data[-9:-3]) * 100
    f.close()
    results[i] = best
    if best > max_result:
        max_result = best
        max_name = file

results = np.around(results.reshape((9,9), order='F'),2)
print('Max Performance:', max_result)
print('Log name:', max_name)

df = pd.DataFrame(results)
df.to_excel(path_name + '.xlsx')