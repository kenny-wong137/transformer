'''
My own transformer implementation.

Main differences from the example implementation in the Tensorflow tutorial:
- The positional embeddings are completely learnable.
  (This is likely to work well on short sequences, but perhaps less so on longer sequences.)
- In the attention query/key vector depth and the number of attention heads can be chosen
  independently of the overall model dimension.
- The value vectors for attention are kept at the model dimension.
- Relu activations are applied wherever possible, including to the attention output,
  and to the feed-forward output.
'''

import tensorflow as tf

class WordAndPositionalEmbedding(tf.keras.layers.Layer):
    
    def __init__(self, model_dims, vocab_size):
        super(WordAndPositionalEmbedding, self).__init__()
        self.word_embed_layer = tf.keras.layers.Embedding(vocab_size, model_dims)
        self.model_dims = model_dims
    
    def build(self, input_shape):
        sequence_len = input_shape[1]
        self.positional_embedding = tf.Variable(
                tf.random.uniform(minval=-0.05, maxval=0.05, shape=(sequence_len, self.model_dims)))
        
    def call(self, inputs):
        word_embedding = self.word_embed_layer(inputs)
        return word_embedding + self.positional_embedding


class AttentionHead(tf.keras.layers.Layer):
    
    def __init__(self, model_dims, attention_depth, causal):
        super(AttentionHead, self).__init__()
        self.sqrt_depth = tf.Variable(tf.math.sqrt(tf.cast(attention_depth, 'float32')),
                                      trainable=False)
        self.query_dense = tf.keras.layers.Dense(attention_depth)
        self.key_dense = tf.keras.layers.Dense(attention_depth)
        self.attention_layer = tf.keras.layers.Attention(causal=causal)
        self.output_dense = tf.keras.layers.Dense(model_dims, activation='relu')

    def call(self, inputs):
        query_inputs = inputs[0]
        key_value_inputs = inputs[1]
        queries = self.query_dense(query_inputs) / self.sqrt_depth
        keys = self.key_dense(key_value_inputs)
        attention_output = self.attention_layer([queries, key_value_inputs, keys])
        return self.output_dense(attention_output)


class MultiHeadAttentionUnit(tf.keras.layers.Layer):
    
    def __init__(self, model_dims, attention_depth, causal, num_heads):
        super(MultiHeadAttentionUnit, self).__init__()
        self.attention_heads = [AttentionHead(model_dims, attention_depth, causal)
                                for _ in range(num_heads)]
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.norm = tf.keras.layers.LayerNormalization()
    
    def call(self, inputs, training):
        query_inputs = inputs[0]
        key_value_inputs = inputs[1]
        attention_outputs = [head([query_inputs, key_value_inputs]) for head in self.attention_heads]
        attention_output_sum = tf.add_n(attention_outputs)
        return self.norm(query_inputs + self.dropout(attention_output_sum, training=training))


class FeedForwardUnit(tf.keras.layers.Layer):
    
    def __init__(self, model_dims, hidden_dims):
        super(FeedForwardUnit, self).__init__()
        self.hidden_dense = tf.keras.layers.Dense(hidden_dims, activation='relu')
        self.output_dense = tf.keras.layers.Dense(model_dims, activation='relu')
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.norm = tf.keras.layers.LayerNormalization()
    
    def call(self, inputs, training):
        hidden_values = self.hidden_dense(inputs)
        outputs = self.output_dense(hidden_values)
        return self.norm(inputs + self.dropout(outputs, training=training))


class EncoderLayer(tf.keras.layers.Layer):
    
    def __init__(self, model_dims, attention_depth, num_heads, hidden_dims):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttentionUnit(model_dims, attention_depth, False, num_heads)
        self.feed_forward = FeedForwardUnit(model_dims, hidden_dims)
    
    def call(self, inputs, training):
        self_attention_output = self.self_attention([inputs, inputs], training=training)
        return self.feed_forward(self_attention_output, training=training)
    

class DecoderLayer(tf.keras.layers.Layer):
    
    def __init__(self, model_dims, attention_depth, num_heads, hidden_dims):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttentionUnit(model_dims, attention_depth, True, num_heads)
        self.cross_attention = MultiHeadAttentionUnit(model_dims, attention_depth, False, num_heads)
        self.feed_forward = FeedForwardUnit(model_dims, hidden_dims)
    
    def call(self, inputs, training):
        decoder_inputs = inputs[0]
        encoder_outputs = inputs[1]
        self_attention_output = self.self_attention([decoder_inputs, decoder_inputs],
                                                    training=training)
        cross_attention_output = self.cross_attention([self_attention_output, encoder_outputs],
                                                      training=training)
        return self.feed_forward(cross_attention_output, training=training)
    

class Encoder(tf.keras.models.Model):
    
    def __init__(self, vocab_size, num_layers, model_dims, attention_depth, num_heads, hidden_dims):
        super(Encoder, self).__init__()
        self.embedding_layer = WordAndPositionalEmbedding(model_dims, vocab_size)
        self.transformer_layers = [EncoderLayer(model_dims, attention_depth, num_heads, hidden_dims)
                                   for _ in range(num_layers)]
    
    def call(self, inputs, training):
        values = self.embedding_layer(inputs)
        for layer in self.transformer_layers:
            values = layer(values, training=training)
        return values


class Decoder(tf.keras.models.Model):
    
    def __init__(self, vocab_size, num_layers, model_dims, attention_depth, num_heads, hidden_dims):
        super(Decoder, self).__init__()
        self.embedding_layer = WordAndPositionalEmbedding(model_dims, vocab_size)
        self.transformer_layers = [DecoderLayer(model_dims, attention_depth, num_heads, hidden_dims)
                                   for _ in range(num_layers)]
    
    def call(self, inputs, training):
        decoder_inputs = inputs[0]
        encoder_outputs = inputs[1]
        decoder_values = self.embedding_layer(decoder_inputs)
        for layer in self.transformer_layers:
            decoder_values = layer([decoder_values, encoder_outputs], training=training)
        return decoder_values


class Transformer(tf.keras.models.Model):
    '''Transformer model'''
    
    def __init__(self, encoder_vocab_size, decoder_vocab_size,
                 num_targets=None, output_activation='softmax',
                 encoder_num_layers=2, decoder_num_layers=2,
                 model_dims=128, attention_depth=16, num_heads=4, hidden_dims=128):
        '''
        :param encoder_vocab_size: number of distinct tokens in encoder input
        :param decoder_vocab_size: number of distinct tokens in decoder input
        :param num_targets: number of distinct target tokens (default=decoder_vocab_size)
        :param output_activation: activation to apply to final output (default='softmax')
        :param encoder_num_layers: number of transformer layers in encoder (default=2)
        :param decoder_num_layers: number of transformer layers in decoder (default=2)
        :param model_dims: the embedding size, also the output size of the transformer layers
        :param attention_depth: the size of a query vector or key vector in an attention unit
        :param num_heads: the number of attention heads in each each attention unit
        :param hidden_dims: the size of the hidden layer in the feed-foward units
        '''
        super(Transformer, self).__init__()
        self.encoder = Encoder(encoder_vocab_size, encoder_num_layers,
                               model_dims, attention_depth, num_heads, hidden_dims)
        self.decoder = Decoder(decoder_vocab_size, decoder_num_layers,
                               model_dims, attention_depth, num_heads, hidden_dims)
        if num_targets is None:
            num_targets = decoder_vocab_size
        self.final_layer = tf.keras.layers.Dense(num_targets, activation=output_activation)
    
    def call(self, inputs, training):
        '''
        :param inputs:
            inputs[0]: encoder inputs, shape=(batch_size, encoder_sequence_len), dtype=int
                                       values between 0 and (encoder_vocab_size - 1)
            inputs[1]: decoder inputs, shape=(batch_size, decoder_sequence_len), dtype=int
                                       values between 0 and (decoder_vocab_size - 1)
        :param training: whether to use dropout in training mode or test mode
        :returns: predictions, shape=(batch_size, dec_sequence_len, num_targets), dtype=float
                  e.g. if output_activation='softmax' then these are the target probabilities
        '''
        encoder_inputs = inputs[0]
        decoder_inputs = inputs[1]
        encoder_outputs = self.encoder(encoder_inputs, training=training)
        decoder_outputs = self.decoder([decoder_inputs, encoder_outputs], training=training)
        return self.final_layer(decoder_outputs)
