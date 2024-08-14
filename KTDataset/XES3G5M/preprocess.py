import pandas as pd


def process_knowledge_tracing_csv(file_path):
    # Read the CSV file with proper handling of quotes and column names
    df = pd.read_csv(file_path, sep=',', quotechar='"')

    # Initialize list to hold processed data
    processed_data = []

    # Process each row
    for index, row in df.iterrows():
        uid = row['uid']
        questions = row['questions'].split(',')
        concepts = row['concepts'].split(',')
        responses = row['responses'].split(',')

        ''' copy多skill情况
        new_questions = []
        new_concepts = []
        new_responses = []

        for q, c, r in zip(questions, concepts, responses):
            split_concepts = c.split('_')
            for sc in split_concepts:
                new_questions.append(int(q))
                new_concepts.append(int(sc))
                new_responses.append(int(r))

        num_questions = len(new_questions)

        processed_data.append([num_questions, new_questions, new_concepts, new_responses])
        '''
        # 保留36_655情况
        questions = list(map(int, questions))
        responses = list(map(int, responses))

        num_questions = len(questions)

        processed_data.append([num_questions, questions, concepts, responses])

    # Prepare the output format
    output_data = []
    for data in processed_data:
        num_questions, questions, concepts, responses = data
        output_data.append(str(num_questions))
        output_data.append(','.join(map(str, questions)))
        output_data.append(','.join(map(str, concepts)))
        output_data.append(','.join(map(str, responses)))

    # Write to the new CSV file
    output_file_path = 'original_train_200.csv'
    with open(output_file_path, 'w') as f:
        for line in output_data:
            f.write(line + '\n')

    return output_file_path


# Process the CSV file
file_path = './train_valid_sequences_quelevel.csv'
output_file = process_knowledge_tracing_csv(file_path)

