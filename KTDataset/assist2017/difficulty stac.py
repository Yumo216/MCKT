import itertools
from collections import defaultdict
import json
# 文件路径
data_file = '../assist2017/assist2017_pid_train.csv'

# 存储题目和回答正确率
question_correct = defaultdict(int)
question_total = defaultdict(int)

def calculate_difficulty(data_file):
    # 读取和处理文件
    with open(data_file, 'r') as file:
        for lens, ques, skill, ans in itertools.zip_longest(*[file] * 4):
            lens = int(lens.lstrip('﻿').strip().strip(','))
            ques = [int(q) for q in ques.strip().strip(',').split(',')]
            ans = [int(a) for a in ans.strip().strip(',').split(',')]

            # 遍历该学生的答题情况，更新每个题目的总答题次数和正确次数
            for q_id, correct in zip(ques, ans):
                question_total[q_id] += 1
                question_correct[q_id] += correct

    # 计算每个题目的正确率
    question_accuracy = {}
    for q_id in question_total:
        if question_total[q_id] < 3:
            question_accuracy[q_id] = 0.5  # 回答数量少于3，设置困难度为0.5
        else:
            question_accuracy[q_id] = question_correct[q_id] / question_total[q_id]
    return question_accuracy

# 调用函数并保存结果到 JSON 文件
question_accuracy = calculate_difficulty(data_file)

with open('XES_diff.json', 'w') as json_file:
    json.dump(question_accuracy, json_file, indent=4)

# 打印结果
for q_id, acc in sorted(question_accuracy.items()):
    print(f"Question ID: {q_id}, Accuracy: {acc:.5f}")