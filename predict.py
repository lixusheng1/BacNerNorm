from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
import tensorflow as tf
import os
import nltk

def main(predict_file,save_file):
    # create instance of config
    config = Config()
    predict=CoNLLDataset(predict_file, config.processing_word, config.max_iter)
    max_sequence_length = max([len(seq[0]) for seq in predict])
    max_word_length = max([len(word[0]) for seq in predict for word in seq[0]])
    print(max_word_length, max_sequence_length)
    model = NERModel(config, max_word_length, max_sequence_length)
    model.build()
    model.restore_session(config.dir_model)
    model.run_predict(predict,save_file)

if __name__ == "__main__":
    main("data/test_set.txt","predict.txt")

