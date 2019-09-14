'''
Download the text of MiddleMarch from the Gutenberg website.
Tokenise the text, and chop it into 32-word chunks.
Each chunk consists of an (input, target) pair, where the target is shifted
to the right by one word. The eventual goal is to train a model to predict the
next word in a sequence.
'''

import os
import urllib
import re
import tensorflow as tf
import numpy as np

URL = 'http://www.gutenberg.org/cache/epub/145/pg145.txt'
FOLDER = 'data/'
RAW_FILE = os.path.join(FOLDER, 'raw_text.txt')

START_INDEX = 111
END_INDEX = 33314


def download_data():
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)
    if not os.path.exists(RAW_FILE):
        urllib.request.urlretrieve(URL, filename=RAW_FILE)

       
def load_words():
    words = []
    with open(RAW_FILE, 'r') as infile:
        for index, line in enumerate(infile):
            if START_INDEX <= index < END_INDEX:
                line = line.lower().strip('\n')
                line = re.sub('([?.!,"-:;])', r' \1 ', line)
                line = re.sub('[^a-z?.!,"-:;]+', ' ', line)
                for word in line.split():
                    words.append(word)
    return words


def tokenize(words, max_vocab_size, oov_token='<unk>'):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
                    filters='', num_words=max_vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(words)
    tokens = tokenizer.texts_to_sequences(words)
    tokens = [item[0] for item in tokens]
    return tokens, tokenizer


def extract_sequences(tokens, seq_len, stride, batch_size):
    input_seqs = []
    target_seqs = []
    for start_index in range(0, (len(tokens) - seq_len - 1), stride):
        input_seqs.append(tokens[start_index : start_index + seq_len])
        target_seqs.append(tokens[start_index + 1 : start_index + seq_len + 1])
        
    input_dataset = tf.data.Dataset.from_tensor_slices(np.array(input_seqs))
    target_dataset = tf.data.Dataset.from_tensor_slices(np.array(target_seqs))
    combined_dataset = tf.data.Dataset.zip((input_dataset, target_dataset))
    return combined_dataset.shuffle(10000).batch(batch_size, drop_remainder=True)


def get_data(max_vocab_size, seq_len, stride, batch_size):
    '''
    Prepare tokenised data.
    
    :param max_vocab_size: maximum vocabulary size
    :param seq_len: length of training sequences
    :param stride: offset between successive sequences
    :param batch_size: batch size
    :returns: dataset (generating batches of tokenised data),
              tokenizer
    '''
    download_data()
    words = load_words()
    tokens, tokenizer = tokenize(words, max_vocab_size)
    dataset = extract_sequences(tokens, seq_len, stride, batch_size)
    return dataset, tokenizer
