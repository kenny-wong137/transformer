Transformer model
========

This repo contains my custom implementation of the [transformer](https://arxiv.org/abs/1706.03762) network architecture in Tensorflow.

Mostly the architecture is standard, but there are a few ways in which the implementation differs from what is presented in the [Tensorflow tutorial](https://www.tensorflow.org/beta/tutorials/text/transformer):

* In my implementation, the positional embeddings are completely learnable. (This is likely to work well on short sequences, but perhaps less so on longer sequences.)
* In my attention units, the query and key vector depths and the number of attention heads can be chosen independently.
* While the query and key vectors are projected to a low-dimensional vector space, the value vectors are kept at their natural dimensions.
* My implementation applies non-linear activations wherever possible. In particularly, relu activations are applied to the outputs of the attention units and feed-forward units.

NB You need Tensorflow 2.0 to run this code.


Applications
-------

**1) Machine translation.** We use the transformer to translate from Spanish to English. The training data consists of paired sentences, downloaded from the
[Tensorflow resources](http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip). The architecture is the standard transformer architecture,
consisting of an encoder portion (which encodes the original Spanish sentence)
together with a decoder portion (which generates the English sentence, word by word, using as context the original Spanish sentence as well as the previous English words).

**2) Language generation.** We train a transformer to generate free English text, in the style of George Eliot's *MiddleMarch*.
The architecture is simpler. There is no need for an encoder; a decoder is enough. At training time, the model learns to predict the next word in a sequence.
At demo time, it then generates fragments of text, one word at a time.

**3) Sentiment analysis.** The transformer is used to classifier IMDB movie reviews into "positive" and "negative" categories. Here, we add a classification head to the first time step (corresponding to the START token). 


Samples
-------

**1) Machine translation.**

```
original spanish:  <start> tom lo sabia todo . <end>
predicted english: <start> tom knew everything . <end>
actual english:    <start> tom knew everything . <end>

original spanish:  <start> ese perro parece completamente una persona . <end>
predicted english: <start> that dog looks pretty good one . <end>
actual english:    <start> this dog is almost human . <end>

original spanish:  <start> siempre esta preparado . <end>
predicted english: <start> she is always prepared . <end>
actual english:    <start> he is always prepared . <end>

original spanish:  <start> un vaso de vino blanco , por favor . <end>
predicted english: <start> a glass of wine , please . <end>
actual english:    <start> a glass of white wine , please . <end>

original spanish:  <start> nunca se es demasiado viejo para aprender . <end>
predicted english: <start> it is never too old to learn . <end>
actual english:    <start> it s never too late to learn . <end>

original spanish:  <start> supongo que tu no puedes hacerlo . <end>
predicted english: <start> i guess that you can t do it . <end>
actual english:    <start> i guess you cannot do it . <end>
```

**2) Language generation.**

[Note that these text fragments do not begin at the start of a sentence or paragraph -- think of them as continuations of existing text.]

```
... mr . frank hawley , lawyer and keeping his brief pause by which he had been allowed to grow
    in bushy beauty and to spread out coral fruit for the birds . little ...

... i wanted to tell bulstrode to bring them into the church under the circumstances ? that
    depends on your conscience , fred - - how far you have counted the cost , as ...

... but when i found it excused lydgate . it ' s pretty nigh two hundred - - there ' s more in
    the box , and nobody knows how much there was . ...

... but kindly - - " look up , nicholas . " he raised his eyes with a little start and looked
    at her half amazed for a moment : her pale face , ...

... a doctor , but if known in the world of reasons crowded upon her against any movement of her
    thought was a poor man , and leaned against the tall back of a ...

... " what is that sir james ? " said dorothea , whose spirits had sunk very low , not only at
    the estimate of his handwriting , but at the vision of himself ...
```


**3) Sentiment analysis**

The accuracy on the validation set reaches a peak of around 88% before the model starts to overfit.

This is comparable to performance with an [LSTM](https://www.tensorflow.org/beta/tutorials/text/text_classification_rnn).
(Though to be fair, the LSTM isn't much better than a [fully-connected network](https://www.tensorflow.org/beta/tutorials/keras/basic_text_classification_with_tfhub). This is because
understanding long-range interactions between words is far less important in this IMDB sentiment analysis problem than computing frequencies of key words.)
