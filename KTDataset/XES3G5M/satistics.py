import pandas as pd

# 假设csv文件名为 data.csv
file_path = 'train_valid_sequences_quelevel.csv'

# 读取CSV文件
df = pd.read_csv(file_path)

# 提取“questions”列中的数据并计算每行的question数量
df['questions_count'] = df['questions'].apply(lambda x: len(x.split(',')))

# 排除 questions 列中的 -1
df['questions'] = df['questions'].apply(lambda x: ','.join([q for q in x.split(',') if q != '-1']))
# 对相同的uid进行合并
df_grouped = df.groupby('uid')['questions'].apply(lambda x: ','.join(x)).reset_index()
# 计算每个uid合并后的question数量
df_grouped['questions_count'] = df_grouped['questions'].apply(lambda x: len(x.split(',')))

# 统计每行最多、最少以及平均的question数量
max_questions = df_grouped['questions_count'].max()
min_questions = df_grouped['questions_count'].min()
average_questions = df_grouped['questions_count'].mean()
median_questions = df_grouped['questions_count'].median()
percentile_95_questions = df_grouped['questions_count'].quantile(0.95)

statistics = {
    'Max Questions': max_questions,
    'Min Questions': min_questions,
    'Average Questions': average_questions,
    'Median Questions': median_questions,
    '95th Percentile Questions': percentile_95_questions,
}
# max_uid = df_grouped[df_grouped['questions_count'] == df_grouped['questions_count'].min()]['uid'].values[0]
print(statistics)