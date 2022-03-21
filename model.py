from tensorflow import keras
from transformers import TFAutoModel, AutoTokenizer
import numpy as np

from database import Database

def get_survey_model(hidden_dim=256, out_dim=128):
    """
    input shape: (102, 56)
    first layer (102, 56) @ (56, 256) -> (102, 256)
    second layer (102, 256) @ (256, 256) -> (102, 256)
    third layer (102, 256) @ (256, 256) -> (102, 256)
    final layer (102, 256) @ (256, 128) -> (102, 128)
    """
    model = keras.Sequential()
    # input: [102, 56]
    model.add(keras.layers.Dense(hidden_dim, activation='relu'))
    # first layer: we have a matrix of shape (56, hidden_dim)
    # output shape: (102, hidden_dim)
    model.add(keras.layers.Dense(hidden_dim, activation='relu'))
    # 2nd layer: we have a matrix of shape (hidden_dim, hidden_dim)
    # output shape: (102, hidden_dim)
    model.add(keras.layers.Dense(hidden_dim, activation='relu'))
    # 3rd layer: we have a matrix of shape (hidden_dim, 512)
    # output shape: (102, 512)
    model.add(keras.layers.Dense(out_dim, activation='relu'))
    # 4th layer: we have a matrix of shape (512, 768)
    # output shape: (102, 768)

    return model


def get_text_model():
    model = TFAutoModel.from_pretrained('distilbert-base-uncased')  # [102, 128]
    return model


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # "I like to eat hambergers" -> [I, like, to, eat, hamburgers] -> [1, 100, 200,]
    return tokenizer


class MusicRecommendationModel(keras.Model):

    def __init__(self, hidden_dim, database=None):
        super().__init__()
        self.text_model = get_text_model()
        self.tokenizer = get_tokenizer()
        self.survey_model = get_survey_model(hidden_dim,
                                             self.text_model.config.hidden_size)
        # what the keras.layers.Dot do:
        # takes two inputs: a, b
        #    a: [N, D]
        #    b: [N, D]
        # returns the dot product of a and b

        # output shape: [N, 1]
        # all the numbers will be between -1 ~ 1
        #    this is because we have normalize=True option

        # this layer calculates cosine similarity of the two input vectors
        # (cosine similarity of survey vector and lyric vector)
        self.similarity = keras.layers.Dot(-1, normalize=True)
        self.database = database

    def call(self, inputs):
        # train_survey = [N, K]
        # train_lyrics = [N, T]
        # 1. putting the survey result into survey model => get survey vector
        survey_vector = self.survey_model(inputs['survey'])  # [102, 128]

        # 2. get the lyric -> put it in text model -> get text vector (embedding)
        lyric_embedding = self.text_model(
            **{k: v for k, v in inputs.items() if k != 'survey'})
        lyric_vector = lyric_embedding.last_hidden_state[:, 0, :]  # [102, 128]

        # 3. check cosine similarity of survey vector and lyric vector
        similarity = self.similarity(
            [lyric_vector,
             survey_vector])  # all the numbers are between -1 ~ 1, shape: [102, 1]
        return similarity

    def recommend(self, survey, recommendations=5):
        if self.database is None:
            print('database not initialized yet')
            return
        survey_vector = self.survey_model(survey).numpy()  # [1, E]
        metadata = self.database.search(survey_vector, topk=recommendations)
        return metadata

    def cache_database(self, all_songs):  #allsongs is pandaa's datadrame
        embeddings = []
        metadata = []
        for i in range(len(all_songs)):  #len(all_lyrics) = N
            song = all_songs.iloc[i]
            lyrics = song['lyrics']
            _metadata = song[['uri', 'name']]  #song uri, song name
            tokenized_lyrics = self.tokenizer(
                lyrics,
                padding=True,
                truncation=True,
                max_length=128,  # the number of words in the lyric
                return_tensors='tf')
            emb = self.text_model(**tokenized_lyrics).last_hidden_state[:, 0, :]
            embeddings.append(
                emb.numpy().squeeze()
            )  #tensorflow to numpy and sqeeuze to remove dimensions and leave as a vector
            metadata.append(_metadata)

        embeddings = np.stack(embeddings, 0)  #[N,768]

        self.database = Database(metadata, embeddings)
        self.database.save('database.pkl')  # save to file