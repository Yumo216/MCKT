import pandas as pd


def process_knowledge_tracing_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, sep=',', quotechar='"')

    # Initialize a dictionary to hold combined data by uid
    combined_data = {}

    # Process each row
    for index, row in df.iterrows():
        uid = row['uid']
        questions = [int(q) for q in row['questions'].split(',') if int(q) != -1]
        concepts = [c for c in row['concepts'].split(',') if c != '-1']
        responses = [int(r) for r in row['responses'].split(',') if int(r) != -1]

        if uid not in combined_data:
            combined_data[uid] = {'questions': [], 'concepts': [], 'responses': []}

        combined_data[uid]['questions'].extend(questions)
        combined_data[uid]['concepts'].extend(concepts)
        combined_data[uid]['responses'].extend(responses)
        '''拆分复制法'''
        # Prepare the processed data with split concepts
        processed_data = []
        for uid, data in combined_data.items():
            num_questions = len(data['questions'])
            new_questions = []
            new_concepts = []
            new_responses = []

            for q, c, r in zip(data['questions'], data['concepts'], data['responses']):
                split_concepts = c.split('_')
                for sc in split_concepts:
                    new_questions.append(q)
                    new_concepts.append(int(sc))
                    new_responses.append(r)

            num_questions = len(new_questions)
            processed_data.append([num_questions, new_questions, new_concepts, new_responses])

    # 保留36_665
    # processed_data = []
    # for uid, data in combined_data.items():
    #     num_questions = len(data['questions'])
    #     processed_data.append([num_questions, data['questions'], data['concepts'], data['responses']])



    # Prepare the output format
    output_data = []
    for data in processed_data:
        num_questions, questions, concepts, responses = data
        output_data.append(str(num_questions))
        output_data.append(','.join(map(str, questions)))
        output_data.append(','.join(map(str, concepts)))
        output_data.append(','.join(map(str, responses)))

    # Write to the new CSV file
    output_file_path = './copy_train.csv'
    with open(output_file_path, 'w') as f:
        for line in output_data:
            f.write(line + '\n')

    return output_file_path


# Process the CSV file
file_path = './train_valid_sequences_quelevel.csv'
output_file_path = process_knowledge_tracing_csv(file_path)



