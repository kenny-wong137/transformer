'''
My own transformer implementation.
This is for a *language model*, so we only have the decoder; there is no encoder.

Main differences from the example implementation in the Tensorflow tutorial:
- The positional embeddings are completely learnable.
  (This is likely to work well on short sequences, but perhaps less so on longer sequences.)
- In the attention units, the query/key vector depth and the number of attention heads
  can be chosen independently of the overall model dimension.
- The value vectors have the same dimension as the model dimension; they are not mapped
  to the dimensionality of the queries and keys.
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
        seq_len = input_shape[1]
        self.positional_embedding = tf.Variable(
                tf.random.uniform(minval=-0.05, maxval=0.05, shape=(seq_len, self.model_dims)))
        
    def call(self, inputs):
        word_embedding = self.word_embed_layer(inputs)
        return word_embedding + self.positional_embedding


class AttentionHead(tf.keras.layers.Layer):
    
    def __init__(self, model_dims, attention_depth):
        super(AttentionHead, self).__init__()
        self.sqrt_depth = tf.Variable(tf.math.sqrt(tf.cast(attention_depth, 'float32')),
                                      trainable=False)
        self.query_dense = tf.keras.layers.Dense(attention_depth)
        self.key_dense = tf.keras.layers.Dense(attention_depth)
        self.attention_layer = tf.keras.layers.Attention(causal=True) # masking the future
        self.output_dense = tf.keras.layers.Dense(model_dims, activation='relu')

    def call(self, inputs):
        query_inputs = inputs[0]
        key_value_inputs = inputs[1]
        queries = self.query_dense(query_inputs) / self.sqrt_depth
        keys = self.key_dense(key_value_inputs)
        attention_output = self.attention_layer([queries, key_value_inputs, keys])
        return self.output_dense(attention_output)


class MultiHeadAttentionUnit(tf.keras.layers.Layer):
    
    def __init__(self, model_dims, attention_depth, num_heads):
        super(MultiHeadAttentionUnit, self).__init__()
        self.attention_heads = [AttentionHead(model_dims, attention_depth)
                                for _ in range(num_heads)]
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.norm = tf.keras.layers.LayerNormalization()
    
    def call(self, inputs, training):
        attention_outputs = [head([inputs, inputs]) for head in self.attention_heads]
        attention_output_sum = tf.add_n(attention_outputs)
        return self.norm(inputs + self.dropout(attention_output_sum, training=training))


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
    
    def __init__(self, model_dims, attention_depth, num_heads, hidden_dims):
        super(TransformerLayer, self).__init__()
        self.self_attention = MultiHeadAttentionUnit(model_dims, attention_depth, num_heads)
        self.feed_forward = FeedForwardUnit(model_dims, hidden_dims)
    
    def call(self, inputs, training):
        attention_output = self.self_attention(inputs, training=training)
        return self.feed_forward(attention_output, training=training)
    

class Transformer(tf.keras.models.Model):
    '''Transformer language model'''
    
    def __init__(self, vocab_size, num_layers, model_dims, attention_depth,
                 num_heads, hidden_dims):
        '''
        :param vocab_size: vocabulary size
        :param num_layers: number of transformer layers
        :param model_dims: embedding dimensionality, also dimensionality of
                           output of transformer layers
        :param attention_depth: sizes of attention queries and keys
        :param num_heads: number of attention heads
        :param hidden_dims: dimensionality of hidden layers in feedforward units
        '''
        super(Transformer, self).__init__()
        self.embedding_layer = WordAndPositionalEmbedding(model_dims, vocab_size)
        self.main_layers = [TransformerLayer(model_dims, attention_depth, num_heads, hidden_dims)
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
