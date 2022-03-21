#%%

import pickle
from database import Database

with open('database.pkl', 'rb') as f:
    database = pickle.load(f)

#%%
database.metadata
#%%
database.embeddings
# %%
with open('metadata.pkl', 'wb') as f:
    pickle.dump(database.metadata, f)
# %%
import numpy as np
np.save('embeddings.npy', database.embeddings)
# %%
