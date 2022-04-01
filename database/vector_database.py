import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class Database:

    def __init__(self, metadata, embeddings):
        self.metadata = metadata
        self.embeddings = embeddings

    def search(self, vector, topk=5):
        # TODO: current implementation is a naive for loop approach.
        similarities = []
        for instance in self.embeddings:
            similarities.append(
                cosine_similarity(vector.reshape(1, -1), instance.reshape(1, -1)))

        similarities = np.array(similarities).squeeze()
        indices = similarities.argsort()[::-1]  # highest sim to lowest

        songs = []
        for idx in indices[:topk]:
            songs.append(self.metadata[idx])

        return songs

    def save(self, filename):
        #in python, there is something called pickle, save virtually anything
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        #save the database into somewhere