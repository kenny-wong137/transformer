'''
My own transformer implementation.

We build three models:
(a) A *language generation model*. Here, we only have a decoder; there is no encoder.
(b) A *classifier*. This adds a classification head onto the 0th time step of the transformer.
(c) A *translator*. This has both a decoder and an encoder.

Main differences from the example implementation in the Tensorflow tutorial:
- The positional embeddings are completely learnable.
  (This is likely to work well on short sequences, but perhaps less so on longer sequences.)
- In the attention units, the query/key vector depth and the number of attention heads
  can be chosen independently of the overall model dimension.
- The value vectors have the same dimension as the model dimension; they are not mapped
  down to the dimensionality of the queries and keys.
- Relu activations are applied to the attention output and to the feed-forward output.
'''

import tensorflow as tf

# Shared code.

class WordAndPositionalEmbedding(tf.keras.layers.Layer):
    
    def __init__(self, model_dims, vocab_size):
        super(WordAndPositionalEmbedding, self).__init__()
        self.word_embed_layer = tf.keras.layers.Embedding(vocab_size, model_dims)
        self.model_dims = model_dims
    
    def build(self, input_shape):
        seq_len = input_shape[1]
        self.positional_embedding = tf.Variable(
                tf.random.uniform(minval=-0.05, maxval=0.05, shape=(seq_len, self.model_dims)))
        # so the positional embedding is fully learnable
        
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
    

class TransformerLayer(tf.keras.layers.Layer):
    
    def __init__(self, model_dims, attention_depth, num_heads, hidden_dims,
                 self_attention_is_causal, has_cross_attention):
        super(TransformerLayer, self).__init__()
        self.self_attention = MultiHeadAttentionUnit(
                    model_dims, attention_depth, self_attention_is_causal, num_heads)
        if has_cross_attention:
            self.cross_attention = MultiHeadAttentionUnit(
                    model_dims, attention_depth, False, num_heads)
        else:
            self.cross_attention = None
        self.feed_forward = FeedForwardUnit(model_dims, hidden_dims)
    
    def call(self, inputs, training):
        if self.cross_attention is not None:
            self_inputs = inputs[0]
            cross_inputs = inputs[1]
        else:
            self_inputs = inputs
        
        values = self.self_attention([self_inputs, self_inputs], training=training)
        if self.cross_attention is not None:
            values = self.cross_attention([values, cross_inputs], training=training)
        return self.feed_forward(values, training=training)


# Definition of language generation model.

class GenerationTransformer(tf.keras.models.Model):
    '''Transformer language model'''
    
    def __init__(self, vocab_size, num_layers, model_dims, attention_depth, num_heads, hidden_dims):
        '''
        :param vocab_size: number of distinct tokens in input
        :param num_layers: number of transformer layers
        :param model_dims: the embedding size, also the output size of the transformer layers
        :param attention_depth: the size of a query vector or key vector in an attention unit
        :param num_heads: the number of attention heads in each each attention unit
        :param hidden_dims: the size of the hidden layer in the feed-foward units
        '''
        super(GenerationTransformer, self).__init__()
        self.embedding_layer = WordAndPositionalEmbedding(model_dims, vocab_size)
        self.main_layers = [TransformerLayer(model_dims, attention_depth, num_heads,
                                             hidden_dims, True, False)
                            for _ in range(num_layers)]
        self.final_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')
    
    def call(self, inputs, training):
        '''
        :param inputs: word sequence inputs
                       shape=(batch_size, seq_len), dtype=int32, maxval=vocab_size
        :param training: whether to run dropout in training mode
        :returns: probabilities for next word
                  shape=(batch_size, seq_len, vocab_size), dtype=float32
        '''
        values = self.embedding_layer(inputs)
        for layer in self.main_layers:
            values = layer(values, training=training)
        return self.final_layer(values)


# Definition of sentiment analysis model.
        
class ClassificationTransformer(tf.keras.models.Model):
    
    def __init__(self, vocab_size, num_layers, model_dims, attention_depth, num_heads, hidden_dims):
        '''
        :param vocab_size: number of distinct tokens in input
        :param num_layers: number of transformer layers
        :param model_dims: the embedding size, also the output size of the transformer layers
        :param attention_depth: the size of a query vector or key vector in an attention unit
        :param num_heads: the number of attention heads in each each attention unit
        :param hidden_dims: the size of the hidden layer in the feed-foward units
        '''
        super(ClassificationTransformer, self).__init__()
        self.embedding_layer = WordAndPositionalEmbedding(model_dims, vocab_size)
        self.main_layers = [TransformerLayer(model_dims, attention_depth, num_heads,
                                             hidden_dims, False, False)
                            for _ in range(num_layers)]
        self.final_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training):
        '''
        :param inputs: word sequence inputs
                       shape=(batch_size, seq_len), dtype=int32, maxval=vocab_size
        :param training: whether to run dropout in training mode
        :returns: classification probability, shape=(batch_size,), dtype=float32
        '''
        values = self.embedding_layer(inputs)
        for layer in self.main_layers:
            values = layer(values, training=training)
        return tf.squeeze(self.final_layer(values[:, 0, :]))


# Definition of translator.

class Encoder(tf.keras.models.Model):
    
    def __init__(self, vocab_size, num_layers, model_dims, attention_depth, num_heads, hidden_dims):
        super(Encoder, self).__init__()
        self.embedding_layer = WordAndPositionalEmbedding(model_dims, vocab_size)
        self.transformer_layers = [TransformerLayer(model_dims, attention_depth, num_heads,
                                                    hidden_dims, False, False)
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
        self.transformer_layers = [TransformerLayer(model_dims, attention_depth, num_heads,
                                                    hidden_dims, True, True)
                                   for _ in range(num_layers)]
    
    def call(self, inputs, training):
        decoder_inputs = inputs[0]
        encoder_outputs = inputs[1]
        decoder_values = self.embedding_layer(decoder_inputs)
        for layer in self.transformer_layers:
            decoder_values = layer([decoder_values, encoder_outputs], training=training)
        return decoder_values


class TranslationTransformer(tf.keras.models.Model):
    '''Transformer model for machine translation'''
    
    def __init__(self, encoder_vocab_size, decoder_vocab_size,
                 encoder_num_layers, decoder_num_layers,
                 model_dims, attention_depth, num_heads, hidden_dims):
        '''
        :param encoder_vocab_size: number of distinct tokens in encoder input
        :param decoder_vocab_size: number of distinct tokens in decoder input
        :param encoder_num_layers: number of transformer layers in encoder
        :param decoder_num_layers: number of transformer layers in decoder
        :param model_dims: the embedding size, also the output size of the transformer layers
        :param attention_depth: the size of a query vector or key vector in an attention unit
        :param num_heads: the number of attention heads in each each attention unit
        :param hidden_dims: the size of the hidden layer in the feed-foward units
        '''
        super(TranslationTransformer, self).__init__()
        self.encoder = Encoder(encoder_vocab_size, encoder_num_layers,
                               model_dims, attention_depth, num_heads, hidden_dims)
        self.decoder = Decoder(decoder_vocab_size, decoder_num_layers,
                               model_dims, attention_depth, num_heads, hidden_dims)
        self.final_layer = tf.keras.layers.Dense(decoder_vocab_size, activation='softmax')
    
    def call(self, inputs, training):
        '''
        :param inputs:
            inputs[0]: encoder inputs
                       shape=(batch_size, encoder_seq_len), dtype=int32, maxval=encoder_vocab_size
            inputs[1]: decoder inputs
                       shape=(batch_size, decoder_seq_len), dtype=int32, maxval=decoder_vocab_size
        :param training: whether to use dropout in training mode or test mode
        :returns:  probabilities for next word
                   shape=(batch_size, decoder_seq_len, decoder_vocab_size), dtype=float32
        '''
        encoder_inputs = inputs[0]
        decoder_inputs = inputs[1]
        encoder_outputs = self.encoder(encoder_inputs, training=training)
        decoder_outputs = self.decoder([decoder_inputs, encoder_outputs], training=training)
        return self.final_layer(decoder_outputs)
