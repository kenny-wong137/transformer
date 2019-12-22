Transformer model
========

This repo contains my custom implementation of the [transformer](https://arxiv.org/abs/1706.03762) network architecture in Tensorflow 2.0,
with applications to machine translation, language generation and sentiment analysis.


**1) Machine translation**

We use the transformer to translate from Spanish to English. The training data consists of paired sentences, downloaded from the
[Tensorflow resources](http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip). The architecture is the standard transformer architecture,
consisting of an encoder portion (which encodes the original Spanish sentence)
together with a decoder portion (which generates the English sentence, word by word, using as context the original Spanish sentence as well as the previous English words).

Below are some examples of translations produced by this network.
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

I would say that these translations are of a similar quality to the translations I produced
in an [earlier exercise](https://github.com/kenny-wong137/deep-learning-exercises/tree/dev/attention)
using an LSTM plus cross-attention.


**2) Language generation**

We train a transformer to generate free English text, in the style of George Eliot's *MiddleMarch*.
The architecture is simpler. There is no need for an encoder; a decoder is enough. At training time, the model learns to predict the next word in a sequence.
At demo time, it then generates fragments of text, one word at a time.

Here are some examples of generated text fragments.
Note that these text fragments do not begin at the start of a sentence or paragraph - think of them as continuations of existing text.
```
... a one point on his rounds . he now imagined itself in a long way here
    before he reopened the sad subject , looking at ...
	
... why , he should not himself have done about you . ” “ oh , he ! ” said
    mrs . bulstrode , with a ...

... a resistant pain , and made her feel keenly it was most inclined to be
    quite inconsistency . it would be admitted to whom he ...
	
... mr . casaubon s feelings were stirred by an air of repose in his mind ,
    and that he would await new duties : many ...

... the top of the carriage , when they went into the curate s pew before
    the door opened and the next day before when he ...

... after celia left her table . she laid her hand on his shoulder , and
    rubbing his hands folded on the opposite sides of his ...

... if mr . farebrother could have taken her and talked persistently . yet
    they were both seated herself , and when she was in a ...
```


**3) Sentiment analysis**

The transformer is used to classifier IMDB movie reviews into "positive" and "negative" categories.
Here, we add a classification head to the first time step (corresponding to the START token). 

The accuracy on the validation set reaches a peak of around 88% before the model starts to overfit.

This is comparable to performance with an [LSTM](https://www.tensorflow.org/beta/tutorials/text/text_classification_rnn).
(Though to be fair, the LSTM isn't much better than a [fully-connected network](https://www.tensorflow.org/beta/tutorials/keras/basic_text_classification_with_tfhub).
This is because understanding long-range interactions between words is far less important in this IMDB sentiment analysis problem than computing frequencies of key words.)

Here is an example of a review, with its model prediction.
```
Prediction = 0.999. Actual label = True

i just viewed eddie monroe and i was very impressed the story was easily paced as the plot
to a surprise ending heartwarming performances action humor and drama filled the screen
acting by some talented long great script this is the best film that fred carpenter has
made to date he should be very proud of this work doug score is on the mark craig morris
is the next tom cruise or brad pitt hard to believe this is a low budget independent film
just imagine what carpenter can do with a hollywood level budget paul last film and he is
greatly missed and loved by all he was a wonderful successful talented actor and a great
human being he will watch over us all and we will never forget his dynamic smile and spirit
great job to all who in this film a few scene stealing and humorous cameos to break up the
serious content you will enjoy this film go see it
```


Implementation details
--------

Mostly the architecture is standard, but there are a few ways in which the implementation differs from what is presented in the
[Tensorflow tutorial](https://www.tensorflow.org/beta/tutorials/text/transformer):

* In my implementation, the positional embeddings are completely learnable. (This is likely to work well on short sequences, but perhaps less so on longer sequences.)
* In my attention units, the query and key vector depths and the number of attention heads can be chosen independently.
* While the query and key vectors are projected to a low-dimensional vector space, the value vectors are kept at their natural dimensions.
* My implementation applies non-linear activations wherever possible. In particularly, relu activations are applied to the outputs of the attention units and feed-forward units.
