'''Trains language model on the text of Middlemarch and simulates fake text.'''

from transformer import GenerationTransformer

import tensorflow as tf
import numpy as np
import os
import urllib
import re

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


def extract_sequences(tokens, seq_len, stride, batch_size, num_epochs):
    input_seqs = []
    target_seqs = []
    for start_index in range(0, (len(tokens) - seq_len - 1), stride):
        input_seqs.append(tokens[start_index : start_index + seq_len])
        target_seqs.append(tokens[start_index + 1 : start_index + seq_len + 1])
        
    input_dataset = tf.data.Dataset.from_tensor_slices(np.array(input_seqs))
    target_dataset = tf.data.Dataset.from_tensor_slices(np.array(target_seqs))
    combined_dataset = tf.data.Dataset.zip((input_dataset, target_dataset))
    return combined_dataset.shuffle(10000).batch(batch_size, drop_remainder=True).repeat(num_epochs)


def get_data(max_vocab_size, seq_len, stride, batch_size, num_epochs):
    download_data()
    words = load_words()
    tokens, tokenizer = tokenize(words, max_vocab_size)
    dataset = extract_sequences(tokens, seq_len, stride, batch_size, num_epochs)
    return dataset, tokenizer


@tf.function
def train_step(model, loss_obj, optimizer, inputs, targets):
    with tf.GradientTape() as tape:
        probs = model(inputs, training=True)
        loss_val = loss_obj(targets, probs)
    gradients = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    

def demo_step(model, tokenizer, possible_first_words, seq_len):
    first_word = np.random.choice(possible_first_words)
    first_token = tokenizer.word_index[first_word]
    sequence = [first_token] + [0 for _ in range(seq_len)]

    for index in range(seq_len):
        inputs = np.array([sequence[:-1]])
        probs = model(inputs, training=False)
        probs = probs.numpy()[0, index, :]
        next_token = np.random.choice(a=len(probs), p=probs)
        sequence[index + 1] = next_token
    
    words = tokenizer.sequences_to_texts([sequence])
    return words[0]


def main(max_vocab_size, seq_len, stride, batch_size, num_epochs, 
         num_layers, model_dims, attention_depth, num_heads, hidden_dims,
         num_batches_per_demo, possible_first_words):
    dataset, tokenizer = get_data(max_vocab_size, seq_len, stride, batch_size, num_epochs)
    '''
    :param max_vocab_size: maximum vocabulary size
    :param seq_len: length of sequences (will chop up the text to make training manageable)
    :param stride: offset between successive sequences
    :param batch_size: batch size
    :param num_epochs: number of complete passes through the dataset
    :param num_layers: number of transformer layers
    :param model_dims: embedding dimension in transformer
    :param attention_depth: depth of attention heads
    :param num_heads: number of heads per attention unit
    :param hidden_dims: number of hidden dimensions in transformer
    :param num_batches_per_demo: frequency at which we generate samples from model
    :param possible_first_words: list of seed words used for generating samples
    '''
    vocab_size = min(len(tokenizer.word_index) + 1, max_vocab_size + 2)
    
    model = GenerationTransformer(vocab_size, num_layers, model_dims,
                                  attention_depth, num_heads, hidden_dims)
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    for batch_index, batch in enumerate(dataset):
        # Train
        inputs, targets = batch
        train_step(model, loss_obj, optimizer, inputs, targets)
        
        # Demo
        if (batch_index + 1) % num_batches_per_demo == 0:
            gen_string = demo_step(model, tokenizer, possible_first_words, seq_len)
            print('Batch {}:\n{}\n'.format(batch_index + 1, gen_string))


if __name__ == '__main__':
    main(max_vocab_size=10000,
         seq_len=32,
         stride=8,
         batch_size=64,
         num_epochs=50,
         num_layers=2,
         model_dims=128,
         attention_depth=16,
         num_heads=4,
         hidden_dims=128,
         num_batches_per_demo=100,
         possible_first_words=['a', 'the', 'i', 'if', 'but', 'why' , '"', 'after', 'his', 'mr']
    )
