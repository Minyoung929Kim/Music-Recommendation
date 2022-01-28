from dataclass import DataClass
import numpy as np

data_class = DataClass('Gathered data cleaned.csv')

encoded_columns, data_mappings, special_mapping = data_class.process_data()

final_data = np.concatenate(list(encoded_columns.values()) +
                            [data_class.df[data_class.continuous_columns].values],
                            axis=1)

print(final_data)
np.save(final_data, 'final_data.npy')
