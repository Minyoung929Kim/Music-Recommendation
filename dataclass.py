import pandas as pd
import numpy as np


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class DataClass:
    """
    We want to manage the data
    """

    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        # opposite question processing
        self.continuous_opposite_questions = [5, 7, 9, 10, 12, 14, 16]

        # TODO: HW1: change the continuous opposite columns:
        # 5 => 1, 4 => 2 , ...
        # save the result in the self.df
        continuous_opposite_columns = []
        for val in self.continuous_opposite_questions:
            for col in self.df.columns:
                if col.startswith(str(val)):
                    continuous_opposite_columns.append(col)
        self.df[continuous_opposite_columns] = 6 - self.df[continuous_opposite_columns]

        self.continuous_questions = [
            3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 20, 22, 23, 24
        ]
        self.continuous_columns = []
        for col in self.df.columns:
            if col != 'Grade' and int(col[0]) in self.continuous_questions:
                self.continuous_columns.append(col)

        self.categorical_columns = [
            'Grade',
            '1) Where do you live during the semester? Please select all.',
            '2) Recently, how many hours did you sleep each day in a week, on average?',
            '8) How long have you been exercising a week?',
            '18) Do you have any ways to release stress? (e.g. meditation)',
            '19) Do you have someone you can talk to about personal thoughts?',
            '21) Recently, how often do you meet with other people for purposes aside from school?',
        ]
        self.special_columns = [
            '17) Which of the following is likely to cause you to feel stressed? (tick all that apply)',
        ]

    def get_mapping(self, column_name, df):
        # 1. select the column from df
        selected_df = df[column_name]

        # 2. create mapping
        mapping = {}

        for i, value in enumerate(selected_df.unique()):
            mapping[value] = i

        return mapping

    def encoding_data(self, column_name, df, mapping):
        """
        mapping = {a: 1, b: 2, c:3}
        df[column] = [a, c, b, b, a]

        encoded_data = [1, 3, 2, 2, 1]
        one_hot_encoded = [[0, 1, 0, 0]
                        [0, 0, 0, 1]
                        [0, 0, 1, 0]
                        [0, 0, 1, 0]
                        [0, 1, 0, 0]]
        """
        # 1. select the column from df
        selected_df = df[column_name]

        # 3. encode the values with the mapping (string -> number)
        # TODO: HW2: get encoded data: i.e. you want to change the values to the number
        # using the mapping
        encoded_data = []
        for val in selected_df:
            encoded_data.append(mapping[val])

        # 4. We want to put the enocded values back into the df
        df['[encoded] ' + column_name] = encoded_data

        # 5. one-hot encoding
        encoded_data = to_categorical(encoded_data, num_classes=len(mapping))
        return encoded_data

    def handle_continuous_columns(self, columns, df):
        # for scaling the continuous values to be in 0~1
        # TODO: HW 3: for scaling the continuous values to be in 0~1
        # you can simply divide by 5.
        # store the scaled value in the original df
        df[columns] = df[columns] / 5

    ### FOR Q17
    def get_multiple_mapping(self, column_name, df):
        all_types = set()
        for data in df[column_name]:
            data = data.split(',')  # 'a,b,c,d,e,f,g,h'  -> ['a', 'b', 'c']
            # 'a, b, c, d' => ['a', ' b', ' c']
            strip_data = []
            for x in data:
                strip_data.append(x.strip())

            all_types.update(strip_data)

        mapping = {x: i for i, x in enumerate(all_types)}
        return mapping

    def encode_multiple(self, column_name, df, mapping):
        encoded_data = np.zeros([len(df), len(mapping)])

        for i, data in enumerate(df[column_name]):
            data = data.split(',')
            strip_data = []
            for x in data:
                strip_data.append(x.strip())
            for x in strip_data:
                encoded_data[i, mapping[x]] = 1

        return encoded_data

    def get_importance_score(self, filename):
        """
        read a pd.DataFrame from filename
        extract importance scores
        flip the scores, so that 1 is the lowest, and 5 is the highest
        take an average over the scores
        return the importance score for each question
        """
        df = pd.read_csv(filename)
        df['a'] = 6 - df['a']
        df['b'] = 6 - df['b']
        df['c'] = 6 - df['c']

        score_mean = (df['a'] + df['b'] + df['c']) / 3
        return score_mean

    def get_depression_score(self, importance_score):
        scores = []
        for row in self.df.iterrows():
            # row  = (index number (int), row as a pd.Series)
            answers = row[1].values[self.continuous_questions]
            # -1, because we want to get the index of the questions
            importance_scores = importance_score[[
                i - 1 for i in self.continuous_questions
            ]]

            depression_score = answers @ importance_scores
            scores.append(depression_score)

        return np.array(scores)

    def process_data(self):
        data_mappings = {}
        encoded_columns = {}
        for column in self.categorical_columns:
            mapping = self.get_mapping(column, self.df)  # dictionary of {value: integer}
            encoded_data = self.encoding_data(column, self.df,
                                              mapping)  # the one hot encoded matrix
            data_mappings[column] = mapping
            encoded_columns[column] = encoded_data

        self.handle_continuous_columns(self.continuous_columns,
                                       self.df)  # scale between 0~1 dividing by 5

        for special_column in self.special_columns:
            special_mapping = self.get_multiple_mapping(special_column, self.df)
            print(special_mapping)
            encoded_columns[special_column] = self.encode_multiple(
                special_column, self.df, special_mapping)
            data_mappings[special_column] = special_mapping

        return encoded_columns, data_mappings

    def process_test_data(self, df, data_mappings):
        encoded_columns = {}

        # categorical
        for column in self.categorical_columns:
            encoded_data = self.encoding_data(
                column, df, data_mappings[column])  # the one hot encoded matrix
            encoded_columns[column] = encoded_data

        # continuous
        self.handle_continuous_columns(self.continuous_columns,
                                       df)  # scale between 0~1 dividing by 5

        # special (i.e. multiple choice)
        for special_column in self.special_columns:
            encoded_columns[special_column] = self.encode_multiple(
                special_column, df, data_mappings[column])
            data_mappings[special_column] = data_mappings[column]

        return encoded_columns


# later...

# import DataClass from dataclass

# data_class = DataClass('Gathered data cleaned.csv')
# encoded_columns, data_mappings, special_mapping = data_class.process_data()
