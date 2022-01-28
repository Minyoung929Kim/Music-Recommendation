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

        self.continuous_columns = [col for col in self.df.columns if '1 - ' in col]

        self.categorical_columns = [
            'Grade',
            '2) Recently, how many hours did you sleep each day in a week, on average?',
            '8) How long have you been exercising a week?',
            '18) Do you have any ways to release stress? (e.g. meditation)',
            '19) Do you have someone you can talk to about personal thoughts?',
            '20) Do you think you spent sufficient time with your family over the last few weeks? (Virtual meetings also)',
            '21) Recently, how often do you meet with other people for purposes aside from school?',
            '22) Recently, how satisfied have you been with your lifestyle?',
            '23) To what extent do you think that mental health care education should be considered seriously?',
            '24) I am able to cope with sudden stress.'
        ]
        self.special_column = '17) Which of the following is likely to cause you to feel stressed? (tick all that apply)'

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
        encoded_data = ...

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
        pass

    ### FOR Q17
    def get_multiple_mapping(self, column_name, df):
        all_types = set()
        for data in df[column_name]:
            data = data.split(',')
            all_types.update(data)

        mapping = {x: i for i, x in enumerate(all_types)}
        return mapping

    def encode_multiple(self, column_name, df, mapping):
        encoded_data = np.zeros([len(df), len(mapping)])

        for i, data in enumerate(df[column_name]):
            data = data.split(',')
            for x in data:
                encoded_data[i, mapping[x]] = 1

        return encoded_data

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

        special_mapping = self.get_multiple_mapping(self.special_column, self.df)
        encoded_columns[self.special_column] = self.encode_multiple(
            self.special_column, self.df, special_mapping)
        data_mappings[self.special_column] = special_mapping

        return encoded_columns, data_mappings, special_mapping


# later...

# import DataClass from dataclass

# data_class = DataClass('Gathered data cleaned.csv')
# encoded_columns, data_mappings, special_mapping = data_class.process_data()
