import itertools
from collections import defaultdict
import json

data_file = '../XES3G5M/original_train.csv'

question_correct = defaultdict(int)
question_total = defaultdict(int)

def calculate_difficulty(data_file):
    with open(data_file, 'r') as file:
        for lens, ques, skill, ans in itertools.zip_longest(*[file] * 4):
            lens = int(lens.lstrip('﻿').strip().strip(','))
            ques = [int(q) for q in ques.strip().strip(',').split(',')]
            ans = [int(a) for a in ans.strip().strip(',').split(',')]

            for q_id, correct in zip(ques, ans):
                question_total[q_id] += 1
                question_correct[q_id] += correct

    question_accuracy = {}
    for q_id in question_total:
        if question_total[q_id] < 3:
            question_accuracy[q_id] = 0.5
        else:
            question_accuracy[q_id] = question_correct[q_id] / question_total[q_id]
    return question_accuracy

question_accuracy = calculate_difficulty(data_file)

with open('XES_diff.json', 'w') as json_file:
    json.dump(question_accuracy, json_file, indent=4)

for q_id, acc in sorted(question_accuracy.items()):
    print(f"Question ID: {q_id}, Accuracy: {acc:.5f}")