from dataclass import DataClass
import numpy as np
import json

data_class = DataClass('Gathered data cleaned.csv')

encoded_columns, data_mappings = data_class.process_data()

final_data = np.concatenate(list(encoded_columns.values()) +
                            [data_class.df[data_class.continuous_columns].values],
                            axis=1)

np.save('survey_data.npy', final_data)

fout = open('data_mappings.json', 'w')
json.dump(data_mappings, fout, indent=4, sort_keys=True)
fout.close()
