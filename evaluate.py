from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
from model.dilated_cnn import Dilated_CNNModel
from model.gram_cnn import GRAM_CNNModel
from model.CNN import CNNModel
import sys

def main(args):
    # create instance of config
    config = Config()
    dev = CoNLLDataset(config.filename_dev, config.processing_word,
                       config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)
    test = CoNLLDataset(config.filename_test, config.processing_word,
                        config.processing_tag, config.max_iter)
    max_sequence_length = max(max([len(seq[0]) for seq in train]), max([len(seq[0]) for seq in dev]),
                              max([len(seq[0]) for seq in test]))

    max_word_length = max(max([len(word[0]) for seq in train for word in seq[0]]),
                          max([len(word[0]) for seq in test for word in seq[0]]),
                          max([len(word[0]) for seq in dev for word in seq[0]]))
    print(max_word_length, max_sequence_length)
    if args == "bilstm":
        model = NERModel(config, max_word_length, max_sequence_length)
    elif args == "gram_cnn":
        model = GRAM_CNNModel(config, max_word_length, max_sequence_length)
    elif args == "gate_cnn":
        model = CNNModel(config, max_word_length, max_sequence_length)
    elif args == "id_cnn":
        model = Dilated_CNNModel(config, max_word_length, max_sequence_length)
    model.build()
    model.restore_session(config.dir_model)
    model.evaluate(test)



if __name__ == "__main__":
    args=sys.argv[1]
    main(args)
