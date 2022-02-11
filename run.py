from dataclass import DataClass
import numpy as np
import json

data_class = DataClass('Gathered data cleaned.csv')

encoded_columns, data_mappings = data_class.process_data()

final_data = np.concatenate(list(encoded_columns.values()) +
                            [data_class.df[data_class.continuous_columns].values],
                            axis=1)

np.save('survey_data/survey_data.npy', final_data)

fout = open('survey_data/data_mappings.json', 'w')
json.dump(data_mappings, fout, indent=4, sort_keys=True)
fout.close()

# get importance scores

importance_scores = data_class.get_importance_score('importance.csv')
depression_scores = data_class.get_depression_score(importance_scores)  # [102,]

# TODO: change the threshold
# 17 continuous questions, each with 1~5, and importance is also 1~5
# in theory, the maximum score can be: 5 * 5 * 17 = 425
# in theory, the minimum score can be: 17

depression_threshold = (importance_scores * 3).sum()

depressed_label = depression_scores > 50  # [102,]
np.save('survey_data/depressed_label.npy', depressed_label)
