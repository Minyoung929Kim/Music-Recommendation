# Music-Recommendation

## Data processing

1. prepared the `importance.csv` file and `Gathered data.csv`.
2. open the `lyrics.ipynb` and collected songs and lyrics.
   1. This should give you the `all_songs.csv`
3. Run `python preprocess_data.py` in your terminal.
4. open `process_data.ipynb` file and run all cells.

## Training Process

To use GPU, you should use Google Colab.
1. open train_model.ipynb, and run all cells.
   1. If you are running on google Colab:
      1. Make a zip filed called Data.zip which contains train_data.pkl, train_label.pkl, test_data.pkl, test_label.pkl.
      2. Upload Data.zip to your google drive.
      3. Run all cells.
      4. Download the best checkpoint, and also, database.pkl.
      5. Rename the best checkpoint to survey_best.data-00000-of-00001 and survey_best.index

## Preparing Database

Just run `python database/prepare_database.py`

## Deploy Procedures

1. If you have some modified files, `git add .`, `git commit -m "some message"`.
2. `git push heroku main`