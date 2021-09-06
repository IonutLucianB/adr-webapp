# something like
# model = load_model(path)

import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.config.experimental.set_visible_devices([], 'GPU')

model_path = "models/cnn_fix1"
#model_path = "models/diz_cnn_emb_3g_1"
# Characters to be used
all_characters = [' ', '\t', '\n', '-', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                  'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'â', 'î', 'ă', 'ș', 'ț']
# Dictionary token to index
token_index = dict([(char, i) for i, char in enumerate(all_characters)])
# Dictionary index to token
reverse_token_index = dict((i, char) for i, char in enumerate(all_characters))
# Dictionary token to index
reverse_char_index = dict(
    (i, char) for char, i in token_index.items())

# Batch size for training
#batch_size = 32
# Max sentence length
max_len = 128
# Max number of samples
#max_samples = 137
# Number of characters
num_tokens = len(all_characters)  # Number of tokens

# LOAD MODEL
model = keras.models.load_model(model_path)

def predict_text(text):
    # PREPARE DATA
    test_texts = [text]

    # ENCODE TEST DATA
    encoder_test_data = np.zeros(
        (1, max_len, num_tokens), dtype="float32"
    )
    for i, test_text in enumerate(test_texts):
        for t, char in enumerate(test_text):
            encoder_test_data[i, t, token_index[char]] = 1.0
        encoder_test_data[i, t + 1:, token_index[" "]] = 1.0

    in_encoder = encoder_test_data
    in_decoder = np.zeros(
        (len(in_encoder), max_len, num_tokens),
        dtype='float32')

    in_decoder[:, 0, token_index["\t"]] = 1

    predict = np.zeros(
        (len(in_encoder), max_len),
        dtype='float32')

    for i in range(max_len - 1):
        predict = model.predict([in_encoder, in_decoder])
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
            if reverse_char_index[x] == "\n":
                break
            else:
                decoded.append(reverse_char_index[x])
        decoded_sentence = "".join(decoded)

    return decoded_sentence

from nltk import ngrams
letters = [' ','-','a','b','c','d','e','f','g','h','i','j','k','l',
'm','n','o','p','q','r','s','t','u','v','w','x','y','z']

of_interest = ['a', 'i', 's', 't']

def split_sentence_to_trigrams(sentence):
    trigrams = ngrams(sentence.split(), 3)
    sentences = []
    try:
      for ngram in trigrams:
          sentence_trigram = ' '.join(ngram)
          sentences.append(sentence_trigram)
    except Exception as e:
      return sentences
    return sentences

def rebuild_from_trigrams(trigrams):
    rebuilt_sentence = ""
    if len(trigrams) == 1:
        return trigrams[0]
    for i in range(len(trigrams)):
        splat_trigram = trigrams[i].split(' ')
        if i == 0:
            rebuilt_sentence += splat_trigram[0] + ' '
        rebuilt_sentence += splat_trigram[1] + ' '
        if i == len(trigrams) - 1:
            rebuilt_sentence += splat_trigram[-1]
    rebuilt_sentence = rebuilt_sentence.strip()
    return rebuilt_sentence

def rebuild_sentence(sentence_in, sentence_out):
    index_out = 0
    sentence_rebuilt = ""
    len_out = len(sentence_out)
    for i in range(len(sentence_in)):
        char_crt = sentence_in[i]
        if char_crt.lower() in letters:
            if len_out > index_out:
                if char_crt in of_interest:
                    if char_crt.isupper():
                        sentence_rebuilt += sentence_out[index_out].upper()
                    else:
                        sentence_rebuilt += sentence_out[index_out]
                else:
                    sentence_rebuilt += char_crt
                index_out += 1
            else:
                sentence_rebuilt += char_crt
        else:
            sentence_rebuilt += char_crt
    return sentence_rebuilt

def preprocess(text):
    processed_text = ""
    for char_ in text:
        if char_.lower() in letters:
            processed_text += char_.lower()
    return processed_text

def predict_text_3g(text):
    preprocessed_text = preprocess(text)
    trigrams = split_sentence_to_trigrams(preprocessed_text)
    diac_trigrams = []
    for trigram in trigrams:
        diac_trigrams.append(predict_text(trigram))
    sentence_out = rebuild_from_trigrams(diac_trigrams)
    return sentence_out

