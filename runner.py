'''Main script - runs language model'''

from pipeline import get_data
from transformer import Transformer

import tensorflow as tf
import numpy as np

@tf.function
def train_step(model, loss_obj, optimizer, inputs, targets):
    '''
    Perform training iteration.
    
    :param model: language model
    :param loss_obj: loss object
    :param optimizer: optimiser
    :param inputs: input sequences
    :param targets: target sequences, offset by one step relative to input sequences
    '''
    with tf.GradientTape() as tape:
        probs = model(inputs, training=True)
        loss_val = loss_obj(targets, probs)
    gradients = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    

def demo_step(model, tokenizer, possible_first_words, seq_len):
    '''
    Generate example sequence from model.
    
    :param model: language model
    :param tokenizer: tokenizer
    :param possible_first_words: list of possible initial words for generated sequence
    :param seq_len: length of sequence, must be consistent with training
    :returns: generated string
    '''
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


def main(max_vocab_size, seq_len, stride, batch_size,
         num_layers, model_dims, attention_depth, num_heads, hidden_dims,
         num_epochs, num_batches_per_demo, possible_first_words):
    dataset, tokenizer = get_data(max_vocab_size, seq_len, stride, batch_size)
    
    model = Transformer(max_vocab_size + 2, num_layers, model_dims,
                        attention_depth, num_heads, hidden_dims)
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    
    for epoch in range(num_epochs):
        for batch_index, batch in enumerate(dataset):
            # Train
            inputs, targets = batch
            train_step(model, loss_obj, optimizer, inputs, targets)
            
            # Demo
            if (batch_index + 1) % num_batches_per_demo == 0:
                gen_string = demo_step(model, tokenizer, possible_first_words, seq_len)
                print('Epoch {}, Batch {}:\n{}\n'.format(epoch + 1, batch_index + 1, gen_string))


if __name__ == '__main__':
    main(max_vocab_size=10000,
         seq_len=32,
         stride=8,
         batch_size=64,
         num_layers=2,
         model_dims=128,
         attention_depth=16,
         num_heads=4,
         hidden_dims=128,
         num_epochs=20,
         num_batches_per_demo=100,
         possible_first_words=['a', 'the', 'i', 'if', 'but', 'why' , '"', 'after', 'his', 'mr']
    )
