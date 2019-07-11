
# tweet analyzation

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import string

# install stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# install tweepy - Anaconda command prompt:
# conda install -c conda-forge tweepy

import tweepy
from tweepy import OAuthHandler

# CSV file for data
import csv

# Regular Expressions
import re

# for Sentiment Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

print("Start")

# Generate new set of data, first run should be 1
generateNewData = 1

# TWITTER SETTINGS
twitterName = "realDonaldTrump"
tweetSince = "2019-01-01"
maxTweetNumber = 100

if(generateNewData):
    
    consumer_key    = 'YOUR_DATA_HERE'
    consumer_secret = 'YOUR_DATA_HERE'
    access_token    = 'YOUR_DATA_HERE'
    access_secret   = 'YOUR_DATA_HERE'
     
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
     
    api = tweepy.API(auth)
    
    with open('tweets.csv', 'w', newline='', encoding="utf-8") as outfile:
        csvWriter = csv.writer(outfile)
        
        # header
        csvWriter.writerow(["TWEET"])
        
        i = 0
        
        for tweet in tweepy.Cursor(api.user_timeline,
                                   screen_name= twitterName,
                                   lang="en",
                                   since=tweetSince,
                                   tweet_mode='extended',
                                   q='-filter:replies -filter:retweets',
                                   ).items():
            
            i+=1
            if i == maxTweetNumber:
                break

            cleanTweet = str(tweet.full_text.encode("utf-8", errors='ignore').decode('UTF-8'))
            
            # link removal
            cleanTweet = re.sub("https://\S+", "", cleanTweet)
            
            
            # mention @ removal
            cleanTweet = re.sub(r"@\S+", "", cleanTweet)
            
            # Removal of unnecessary parts
            # b' and b" from beginning and ' " in general
            
            cleanTweet = re.sub("\'", "", cleanTweet)
            cleanTweet = re.sub("\"", "", cleanTweet)
            cleanTweet = re.sub("\´", "", cleanTweet)
            
            # skip comments (starts with whitespaces)
            if re.match(r"\s", cleanTweet) or re.match(r"^RT", cleanTweet):
                maxTweetNumber+=1
                continue
            
            # blank row skip
            if not (cleanTweet and cleanTweet.strip()):
                maxTweetNumber+=1
                continue
            
            cleanTweet = cleanTweet.strip()
            
            csvWriter.writerow([cleanTweet])
            
    outfile.close()

# WORD FREQUENCY ANALYSATION

# inport from file
input_file = "tweets.csv"
data = pd.read_csv(input_file,header=0)

# PREPARING DATA

# lowercase
data['TWEET'] = data['TWEET'].str.lower()

# stopword removal
stop = stopwords.words('english')
data['TWEET'] = data['TWEET'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# punctuation removal
data['TWEET'] = data['TWEET'].apply(lambda x: " ".join(x for x in x.split() if x not in string.punctuation))

# filter non alphabetical and numreical characters
data['TWEET'] = data['TWEET'].apply(lambda x: " ".join(x for x in x.split() if x.isalnum() and not x.isdigit()))

# plot settings
count1 = Counter(" ".join(data['TWEET']).split()).most_common(20)
df1 = pd.DataFrame.from_dict(count1)
df1 = df1.rename(columns={0: "words in twitter messages", 1 : "count"})
y_pos = np.arange(len(df1["words in twitter messages"]))
plt.barh(y_pos, df1["count"])
plt.yticks(y_pos, df1["words in twitter messages"])

# plot styling
plt.title('More frequent words in twitter messages WITHOUT stopwords')
plt.ylabel('words')
plt.xlabel('number')
plt.grid(True)
plt.gca().invert_yaxis()

plt.show()


# SENTIMENT ANALYSIS

# import again, because here we dont use stopword removal
input_file = "tweets.csv"
data = pd.read_csv(input_file,header=0)


# first run
nltk.download('subjectivity')
nltk.download('punkt')
nltk.download('vader_lexicon')

n_instances = 100

subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
len(subj_docs), len(obj_docs)
subj_docs[0]

train_subj_docs = subj_docs[:80]
test_subj_docs = subj_docs[80:100]
train_obj_docs = obj_docs[:80]
test_obj_docs = obj_docs[80:100]
training_docs = train_subj_docs+train_obj_docs
testing_docs = test_subj_docs+test_obj_docs
sentim_analyzer = SentimentAnalyzer()

all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])

unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)

sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)

trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

sid = SentimentIntensityAnalyzer()

for sentence in data['TWEET']:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print("\n")

print('End')
