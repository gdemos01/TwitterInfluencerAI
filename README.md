# Twitter Influencer AI - Kylie Trump
An AI that generates tweets inspired by famous "influencers" using NLP (ft Donald Trump & Kylie Jenner)

### YouTube Video with Process and Results: https://youtu.be/YIvfM_DP918

<img align="left" width="130" src="https://github.com/gdemos01/TwitterInfluencerAI/blob/master/data/KylieTrump.jpg">

### Kylie Trump
 
>Hey guys. I'm Kylie. In my spare time I like creating cosmetics and running for the White House presidency.\
\
>Also, I'm definitely not an AI that uses Deep RNNs 

#### Twitter: https://twitter.com/TrumpKylie


## How to use

### Stream Tweets
Stream tweets with a specific set of hashtags or keywords. In order for this to work you have to replace the dummy Twitter credentials
with your own. You can change the keywords/hashtags through the code

```
python Controller.py --stream_twitter
```

### Train Model
The command bellow trains the model for 100 epochs. Each epoch creates a new checkpoint if the model is improved. You can change the input file through the code.
```
python Controller.py --train_model 100
```

### Generate Tweet
The command bellow generates text with a lenght of 250 characters. It uses "Hello world" as a seed to begin predicting the new text.
The weights of the model are loaded from "checkpoint_8"

```
python Controller.py --generate_text 250 --seed="Hello world" --checkpoint="checkpoint_8"
```

## Resources
* Streaming Tweets Tutorial: https://www.youtube.com/watch?v=wlnx-7cm4Gg
* Twitter API Python Lib: https://tweepy.readthedocs.io/en/v3.5.0/
* Donald Trump Dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FKJEBIL
* Text Generation with Python and TensorFlow/Keras (LSTM): https://stackabuse.com/text-generation-with-python-and-tensorflow-keras/
* Text Generation with Keras and TensorFlow video: https://www.youtube.com/watch?v=6ORnRAz3gnA
* Text Generation Using Tensorflow: https://www.tensorflow.org/tutorials/text/text_generation

