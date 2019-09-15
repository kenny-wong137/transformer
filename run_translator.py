'''Trains Spanish -> English translation model'''

from transformer import TranslationTransformer

import tensorflow as tf
from sklearn.model_selection import train_test_split
import unicodedata
import re
import os
import io
import urllib
from zipfile import ZipFile

# Data prep

URL = 'http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip'
DATA_FOLDER = 'data/'
DATA_ZIP = os.path.join(DATA_FOLDER, 'spa-eng.zip')
DATA_FILE = os.path.join(DATA_FOLDER, 'spa-eng', 'spa.txt')

OOV_TOKEN = '<unk>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'


def download_data():
    if not os.path.exists(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)
    if not os.path.exists(DATA_ZIP):
        urllib.request.urlretrieve(URL, filename=DATA_ZIP)
        with ZipFile(DATA_ZIP, 'r') as zip_obj:
           zip_obj.extractall(path=DATA_FOLDER)


def clean_sentence(sentence):
    sentence = ''.join(char for char in unicodedata.normalize('NFD', sentence.lower().strip())
                       if unicodedata.category(char) != 'Mn')
    sentence = re.sub('([?.!,¿])', r' \1 ', sentence)
    sentence = re.sub('[" "]+', ' ', sentence)
    sentence = re.sub('[^a-zA-Z?.!,¿]+', ' ', sentence)
    sentence = sentence.strip()
    sentence = '{} {} {}'.format(START_TOKEN, sentence, END_TOKEN)
    return sentence


def load_text():
    sp_text = []
    en_text = []
    for line in io.open(DATA_FILE, encoding='UTF-8').read().strip().split('\n'):
        sentences = line.split('\t')
        assert len(sentences) == 2
        en_text.append(clean_sentence(sentences[0]))
        sp_text.append(clean_sentence(sentences[1]))
    return sp_text, en_text


def tokenise(text, max_vocab_size):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
                    filters='', num_words=max_vocab_size, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(text)
    tensor = tokenizer.texts_to_sequences(text)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, tokenizer


def remove_sentences_with_unknown_tokens(sp_data, en_data, sp_tokenizer, en_tokenizer):
    sp_unknown_index = sp_tokenizer.word_index[OOV_TOKEN]
    en_unknown_index = en_tokenizer.word_index[OOV_TOKEN]
    sp_has_unknown = tf.reduce_any(tf.math.equal(sp_data, sp_unknown_index), axis=1)
    en_has_unknown = tf.reduce_any(tf.math.equal(en_data, en_unknown_index), axis=1)
    all_words_known = tf.logical_not(tf.logical_or(sp_has_unknown, en_has_unknown))
    return sp_data[all_words_known], en_data[all_words_known] 


def get_data(max_vocab_size, test_size, batch_size, num_epochs):
    download_data()
    sp_text, en_text = load_text()
    sp_data, sp_tokenizer = tokenise(sp_text, max_vocab_size)
    en_data, en_tokenizer = tokenise(en_text, max_vocab_size)

    sp_data, en_data = remove_sentences_with_unknown_tokens(
                            sp_data, en_data, sp_tokenizer, en_tokenizer)

    sp_train_data, sp_test_data, en_train_data, en_test_data = train_test_split(
            sp_data, en_data, test_size=test_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((sp_train_data, en_train_data))
    train_dataset = train_dataset.shuffle(1000).batch(batch_size).repeat(num_epochs)
    test_dataset = tf.data.Dataset.from_tensor_slices((sp_test_data, en_test_data))
    test_dataset = test_dataset.shuffle(1000).batch(1).repeat(None)
    
    return train_dataset, test_dataset, sp_tokenizer, en_tokenizer


# Training and predicting

@tf.function
def train_step(model, loss_obj, optimizer, sp_real, en_real):
    en_input = en_real[:, :-1]
    en_target = en_real[:, 1:]
    with tf.GradientTape() as tape:
        probs = model([sp_real, en_input], training=True)
        loss_val = loss_obj(en_target, probs)
    gradients = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def to_words(seq, tokenizer):
    seq = seq.numpy()
    non_padding_tokens = [token for token in seq if token != 0]
    words = tokenizer.sequences_to_texts([non_padding_tokens])
    return words[0]


def demo_step(model, sp_real, en_real, sp_tokenizer, en_tokenizer, batch_index):
    assert tf.shape(sp_real).numpy()[0] == 1 and tf.shape(en_real).numpy()[0] == 1
    en_seq_len = tf.shape(en_real).numpy()[1]
    dec_steps = [tf.zeros(shape=(1,), dtype='int32') for _ in range(en_seq_len)]
    dec_steps[0] = tf.constant(en_tokenizer.word_index[START_TOKEN], shape=(1,), dtype='int32')
    
    for step_id in range(en_seq_len - 1):
        dec_inputs = tf.stack(dec_steps[:-1], axis=1)
        probs = model([sp_real, dec_inputs], training=False)
        preds = tf.argmax(probs[:, step_id, :], axis=1)
        dec_steps[step_id + 1] = tf.cast(preds, 'int32')
        
    dec_outputs = tf.stack(dec_steps, axis=1)

    print('Batch {}:'.format(batch_index + 1))
    print('original spanish:  ' + to_words(sp_real[0, :], sp_tokenizer))
    print('predicted english: ' + to_words(dec_outputs[0, :], en_tokenizer))
    print('actual english:    ' + to_words(en_real[0, :], en_tokenizer))
    print('')


def main(max_vocab_size, test_size, batch_size, num_epochs,
         num_layers, model_dims, attention_depth, num_heads, hidden_dims,
         num_batches_per_demo):
    '''
    :param max_vocab_size: maximum vocabulary size; sentences with words outside this
                           vocabulary are removed
    :param test_size: size of holdout validation set
    :param batch_size: batch size for training set (NB batch size for test set is 1)
    :param num_epochs: number of epochs for training set (NB test set repeats forever)
    :param num_layers: number of layers for transformer
    :param model_dims: embedding dimension in transformer
    :param attention_depth: depth of attention heads
    :param num_heads: number of heads per attention unit
    :param hidden_dims: number of hidden dimensions in transformer
    :param num_batches_per_demo: frequency at which we print sample translations from model
    '''
    train_dataset, test_dataset, sp_tokenizer, en_tokenizer = get_data(
            max_vocab_size, test_size, batch_size, num_epochs)
            
    sp_vocab_size = min(len(sp_tokenizer.word_index) + 1, max_vocab_size + 2)
    en_vocab_size = min(len(en_tokenizer.word_index) + 1, max_vocab_size + 2)

    model = TranslationTransformer(sp_vocab_size, en_vocab_size, num_layers, num_layers,
                                   model_dims, attention_depth, num_heads, hidden_dims)
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    test_iterator = iter(test_dataset)

    for (batch_index, (sp_real, en_real)) in enumerate(train_dataset):
        train_step(model, loss_obj, optimizer, sp_real, en_real)
        
        if (batch_index + 1) % num_batches_per_demo == 0:
            sp_real, en_real = next(test_iterator)
            demo_step(model, sp_real, en_real, sp_tokenizer, en_tokenizer, batch_index)


if __name__ == '__main__':
    main(max_vocab_size=2500,
         test_size=0.05,
         batch_size=64,
         num_epochs=8,
         num_layers=2,
         model_dims=128,
         attention_depth=16,
         num_heads=4,
         hidden_dims=128,
         num_batches_per_demo=100
    )
