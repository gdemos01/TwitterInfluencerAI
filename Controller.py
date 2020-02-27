from NLP import NLP
from TwitterAPI import TwitterAPI
import argparse

if __name__ == '__main__':
    print("\n> Welcome to Twitter Influencer AI\n")
    print(  "████████▀▀░░░░░░░░░░░░░░░░░░░▀▀████████"+"\n"+
            "██████▀░░░░░░░░░░░░░░░░░░░░░░░░░▀██████"+"\n"+
            "█████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█████"+"\n"+
            "████░░░░░▄▄▄▄▄▄▄░░░░░░░░▄▄▄▄▄▄░░░░░████"+"\n"+
            "████░░▄██████████░░░░░░██▀░░░▀██▄░░████"+"\n"+
            "████░░███████████░░░░░░█▄░░▀░░▄██░░████"+"\n"+
            "█████░░▀▀███████░░░██░░░██▄▄▄█▀▀░░█████"+"\n"+
            "██████░░░░░░▄▄▀░░░████░░░▀▄▄░░░░░██████"+"\n"+
            "█████░░░░░█▄░░░░░░▀▀▀▀░░░░░░░█▄░░░█████"+"\n"+
            "█████░░░▀▀█░█▀▄▄▄▄▄▄▄▄▄▄▄▄▄▀██▀▀░░█████"+"\n"+
            "██████░░░░░▀█▄░░█░░█░░░█░░█▄▀░░░░██▀▀▀▀"+"\n"+
            "▀░░░▀██▄░░░░░░▀▀█▄▄█▄▄▄█▄▀▀░░░░▄█▀░░░▄▄"+"\n"+
            "▄▄▄░░░▀▀██▄▄▄▄░░░░░░░░░░░░▄▄▄███░░░▄██▄"+"\n"+
            "██████▄▄░░▀█████▀█████▀██████▀▀░░▄█████"+"\n"+
            "██████████▄░░▀▀█▄░░░░░▄██▀▀▀░▄▄▄███▀▄██")


    parser = argparse.ArgumentParser()
    parser.add_argument("--stream_twitter",action="store_true")
    parser.add_argument("--train_model") # input the number of epochs
    parser.add_argument("--generate_text") #input the length of text
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed")
    args = parser.parse_args()

    if(args.stream_twitter):
        access_token = "529412295-hOTmVeR3LB0luZqspro3iYoAvYGQ2VLxBRkkSCwP"
        access_token_secret = "0kwpFyp7y29MaQa8Q6yg6f9DKv9UwkhAswLpC0C5PtCuV"
        api_key = "s9exhF2ZuidZxGj29pngvzNzB"
        api_secret_key = "yzvFcsjYgI748Dp9xFh3lQVwymLdnPJGPovTwMqe7Ggw5b2hQs"

        # Data Collection
        twitter_api = TwitterAPI(access_token, access_token_secret, api_key, api_secret_key)
        #twitter_api.streamTweets(['#InfluencerLife', '#Influencer'])
        twitter_api.streamTweets(
            ['#InfluencerLife', '#Influencer', '#beauty', '#makeup', '#lifestyle', '#fashion', '#instagram'])
    else:
        text = open('data/KyleJenner.txt','rb').read().decode(encoding='utf-8')
        nlp = NLP()
        dataset, vocabulary = nlp.preprocess(text)
        dataset, vocabulary_size, embedding_dimension, rnn_nodes, batch_size = nlp.prepareSettings(dataset,vocabulary)
        if (args.train_model):
            model = nlp.buildModel(vocabulary_size,embedding_dimension,batch_size,rnn_nodes)
            nlp.trainModel(dataset,model,'checkpoints/',int(args.train_model))
        elif (args.generate_text):
            model = nlp.buildModel(vocabulary_size,embedding_dimension,1,rnn_nodes) # Using batch_size=1 for text gen
            print(model.summary())
            generated_text = nlp.generateText(model,'checkpoints/'+args.checkpoint,args.seed,int(args.generate_text))
            print(generated_text)
        else:
            print("\n > No such option \n")