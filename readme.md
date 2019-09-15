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
original spanish:  <start> yo normalmente me doy una ducha antes de ir a la cama . <end>
predicted english: <start> i usually take me a shower before i go to bed . <end>
actual english:    <start> i usually take a bath before going to bed . <end>

original spanish:  <start> el se hizo policia . <end>
predicted english: <start> he became the police . <end>
actual english:    <start> he became a policeman . <end>

original spanish:  <start> ¿ quereis comer algo antes de que nos vayamos ? <end>
predicted english: <start> do you want to eat something before we go ? <end>
actual english:    <start> do you want to eat something before we leave ? <end>

original spanish:  <start> tom es mi amigo . <end>
predicted english: <start> tom is my friend . <end>
actual english:    <start> tom s my friend . <end>

original spanish:  <start> ¿ me puedo comer esa torta ? <end>
predicted english: <start> can i eat that cake ? <end>
actual english:    <start> may i eat this cake ? <end>

original spanish:  <start> queria que tom tuviera una copia de mi nueva novela . <end>
predicted english: <start> i wanted tom to have a copy of new novel . <end>
actual english:    <start> i wanted tom to have a copy of my new novel . <end>
```

**2) Language generation.**

[Note that these text fragments do not begin at the start of a sentence or paragraph -- think of them as continuations of existing text.]

```
... after a set ignorance . the other medical reform would , was in a case with the
       ways of helping to recognize . but in ...

... a negative . mary had felt a resistant emotion . he could hardly give his pain ,
       could be no sooner did or see him ...

... i don ' t know you think that i am sure you would do it for you . " " if you
       will think me ...

... mr . farebrother recurred to her knitting with a dignified satisfaction in her
       neat fashion , but inspiration could hardly have been easy to be ...

... i was not to be sure of being admired . " i await ladislaw , my life would be
       offensive capability of criticism , necessarily ...

... mr . brooke , nodding at dorothea as she said , " but then you will be painted ?
       " when they are the letters ...

... if you think i shall vote for the better to try and make a good deal of trouble .
       don ' t you come to ...
```


**3) Sentiment analysis**

The accuracy on the validation set reaches a peak of around 88% before the model starts to overfit.

This is comparable to performance with an [LSTM](https://www.tensorflow.org/beta/tutorials/text/text_classification_rnn).
(Though to be fair, the LSTM isn't much better than a [fully-connected network](https://www.tensorflow.org/beta/tutorials/keras/basic_text_classification_with_tfhub). This is because
understanding long-range interactions between words is far less important in this IMDB sentiment analysis problem than computing frequencies of key words.)
