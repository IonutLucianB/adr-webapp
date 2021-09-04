import abc
from tensorflow import keras
import numpy as np


class Model(metaclass=abc.ABCMeta):

    ALPHABET = [' ', '\t', '\n', '-', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'â', 'î', 'ă', 'ș', 'ț']

    TOKEN_INDEX = dict([(char, i) for i, char in enumerate(ALPHABET)])

    INDEX_TOKEN = dict([(i, char) for i, char in enumerate(ALPHABET)])

    NUM_TOKENS = len(ALPHABET)

    @abc.abstractmethod
    def predict_text(self):
        pass


class ModelCNN(Model):

    MAX_LEN = 128

    def __init__(self, path):
        self.model = keras.models.load_model(path)

    def predict_text(self, text):
        # PREPARE DATA
        test_texts = [text]

        # ENCODE TEST DATA
        encoder_test_data = np.zeros(
            (1, self.MAX_LEN, self.NUM_TOKENS), dtype="float32"
        )
        for i, test_text in enumerate(test_texts):
            for t, char in enumerate(test_text):
                encoder_test_data[i, t, self.TOKEN_INDEX[char]] = 1.0
            encoder_test_data[i, t + 1:, self.TOKEN_INDEX[" "]] = 1.0

        in_encoder = encoder_test_data
        in_decoder = np.zeros(
            (len(in_encoder), self.MAX_LEN, self.NUM_TOKENS),
            dtype='float32')

        in_decoder[:, 0, self.TOKEN_INDEX["\t"]] = 1

        predict = np.zeros(
            (len(in_encoder), self.MAX_LEN),
            dtype='float32')

        for i in range(self.MAX_LEN - 1):
            predict = self.model.predict([in_encoder, in_decoder])
            predict = predict.argmax(axis=-1)
            predict_ = predict[:, i].ravel().tolist()
            for j, x in enumerate(predict_):
                in_decoder[j, i + 1, x] = 1

        for seq_index in range((len(in_encoder))):
            # Take one sequence (part of the training set)
            # for trying out decoding.
            output_seq = predict[seq_index, :].ravel().tolist()
            decoded = []
            for x in output_seq:
                if self.INDEX_TOKEN[x] == "\n":
                    break
                else:
                    decoded.append(self.INDEX_TOKEN[x])
            decoded_sentence = "".join(decoded)

        return decoded_sentence
