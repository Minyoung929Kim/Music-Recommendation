from dataclass import DataClass
import numpy as np
import json

data_class = DataClass('Gathered data.csv')

encoded_columns, data_mappings = data_class.process_data()

final_data = np.concatenate(list(encoded_columns.values()) +
                            [data_class.df[data_class.continuous_columns].values],
                            axis=1)

np.save('survey_data/survey_data.npy', final_data)

fout = open('survey_data/data_mappings.json', 'w')
json.dump(data_mappings, fout, indent=4, sort_keys=True)
fout.close()

# get importance scores

importance_scores = data_class.get_importance_score('survey_data/importance.csv')
maximum_depression_score = importance_scores[[
    i - 1 for i in data_class.continuous_questions  # [3, 2, 1] -> [2, 1, 0]
]].sum()

depression_scores = data_class.get_depression_score(importance_scores)  # [102,]

# NOTE: decided the threshold to be 14 after the meeting (Feb. 23, 2022)
depression_threshold = (14 / 17) * maximum_depression_score

depressed_label = depression_scores > depression_threshold  # [102,]
np.save('survey_data/depression_label.npy', depressed_label)
