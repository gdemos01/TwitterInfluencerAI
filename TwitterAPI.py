import json
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

class TwitterStreamListener(StreamListener):

    def on_data(self, raw_data):
        try:
            print("\nNew Tweet:\n-------------\n"+json.loads(raw_data)['text']+"\n")
            tweet_data = json.loads(raw_data)
            if "extended_tweet" in tweet_data:
                tweet = tweet_data['extended_tweet']['full_text']
            else:
                tweet = tweet_data['text']
            f = open("tweets.txt", "a",encoding='utf-8')
            f.write(tweet)
            f.close()
            return True
        except Exception as e:
            print("Error on_data: ",e)
        return True

    def on_error(self,status_code):
        print(status_code)

"""
    Streaming Tweets in real time based on a specific set of hashtags or text
"""
class TwitterAPI:

    def __init__(self,access_token,access_token_secret,api_key,api_secret_key):
        print("> Twitter API initialized")
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.api_key = api_key
        self.api_key_secret = api_secret_key

    def streamTweets(self, hashtags):
        print("> Streaming Tweets with hashtags: ",hashtags)

        # Stream Listener
        listener = TwitterStreamListener()

        # Authentication
        auth = OAuthHandler(self.api_key, self.api_key_secret)
        auth.set_access_token(self.access_token,self.access_token_secret)

        # Tweet Stream
        stream = Stream(auth,listener,tweet_mode= 'extended')
        stream.filter(track=hashtags, languages=['en'])
