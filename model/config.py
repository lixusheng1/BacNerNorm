import os
from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word

class Config():
    def __init__(self, load=True):
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
        self.logger = get_logger(self.path_log)
        if load:
            self.load()

    def load(self):
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)
        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)
        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,self.vocab_chars, lowercase=True)
        self.processing_tag  = get_processing_word(self.vocab_tags,lowercase=False, allow_unk=False)
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)if self.use_pretrained else None)


    # general config
    dir_output = "results/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_word = 300
    dim_char = 25

    # glove files
    filename_glove = "data/embedding/word2vec.300d.txt"
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = "data/word2vec.4B.{}d.trimmed.npz".format(dim_word)
    use_pretrained = True

    # dataset
    filename_dev = "data/dev_set.txt"
    filename_test = "data/test_set.txt"
    filename_train = "data/train_set.txt"
    max_iter = None
    filename_words = "data/words.txt"
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"

    # training
    train_embeddings = False
    nepochs          = 100
    dropout          = 0.5
    batch_size       = 1
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1
    nepoch_no_imprv  = 5
    momentum=0.9

    # model hyperparameters
    hidden_size_char = 25
    hidden_size_lstm = 25
    use_crf = True
    use_char_lstm =False
    use_char_cnn=True
    use_attention=""
    #-------------------------------char_cnn----------------------------------------
    filter_size=3
    filter_deep=30
    #---------------------------------gate_cnn---------------------------------------------
    num_layers=3
    cnn_dim=200
    #-----------------------------idcnn----------------------------------------------------
    layers = [{'dilation': 1},{ 'dilation': 2},{ 'dilation': 1 },]
    idcnn_num_filters=200
    idcnn_filter_width=3
    repeats=1
    share_repeats=True
    #------------------------------lm--------------------------------
    hidden_size_lm=100
    lm_hidden_layer_size=50

