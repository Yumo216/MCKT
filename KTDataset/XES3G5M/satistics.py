import pandas as pd


file_path = 'train_valid_sequences_quelevel.csv'


df = pd.read_csv(file_path)
df['questions_count'] = df['questions'].apply(lambda x: len(x.split(',')))
df['questions'] = df['questions'].apply(lambda x: ','.join([q for q in x.split(',') if q != '-1']))
df_grouped = df.groupby('uid')['questions'].apply(lambda x: ','.join(x)).reset_index()
df_grouped['questions_count'] = df_grouped['questions'].apply(lambda x: len(x.split(',')))


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