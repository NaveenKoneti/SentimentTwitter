import tweepy
from tweepy import OAuthHandler
import pandas as pd
import string
import itertools
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import *



consumer_key = 'Enter Your consumer key'
consumer_secret = 'Enter Your Consumer Secret'
access_token = 'Enter Your Access Token'
access_secret = 'Enter Your Access Secret'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

Data = pd.read_csv("Sentiment_Analysis_Dataset_Updated.csv")

positiveData = Data.loc[Data.Sentiment == 'positive'][:20000]
negativeData = Data.loc[Data.Sentiment == 'negative'][:20000]

positiveSubset = positiveData[['SentimentText', 'Sentiment']]

def GetCleanedData(TextDataColumn):
    wordsPos = TextDataColumn.str.split()
    NoApos = wordsPos.map(lambda x: ' '.join(x).replace("'re"," are").replace("'s"," is").replace("n't"," not").replace("'d"," had").replace("'ll"," will").replace("'m"," am").replace("'ve"," have"))
    NoApos = NoApos.map(lambda x: x.translate(None, string.punctuation))
    NoApos = NoApos.map(lambda x: ''.join(''.join(s)[:2] for _,s in itertools.groupby(x)))
    NoApos = NoApos.map(lambda x: re.sub(r"http\S+", "", x))


    NoApos = NoApos.map(lambda x:[x for x in x.lower().split() if x not in NoApos])
    return NoApos

positiveSubset['SentimentText1'] = GetCleanedData(positiveSubset['SentimentText'])
del positiveSubset['SentimentText']
cols = positiveSubset.columns.tolist()
cols = cols[-1:] + cols[:-1]
positiveSubset = positiveSubset[cols]
positiveSubset['SentimentText1'] = positiveSubset['SentimentText1'].map(lambda x: ' '.join(x))


negativeSubset = negativeData[['SentimentText', 'Sentiment']]
negativeSubset['SentimentText1'] = GetCleanedData(negativeSubset['SentimentText'])
del negativeSubset['SentimentText']

cols = negativeSubset.columns.tolist()
cols = cols[-1:]+cols[:-1]
negativeSubset = negativeSubset[cols]
negativeSubset['SentimentText1'] = negativeSubset['SentimentText1'].map(lambda x: ' '.join(x))


XtrainPos = positiveSubset['SentimentText1'].tolist()
Xtrainneg = negativeSubset['SentimentText1'].tolist()
Xtrain = XtrainPos+Xtrainneg

Ytrain = positiveSubset['Sentiment'].tolist()+negativeSubset['Sentiment'].tolist()
Ytrain = [1 if (x == 'positive') else 0 for x in Ytrain]


vectorizer = TfidfVectorizer(min_df=5,
                             max_df = 0.8,
                             sublinear_tf=True,
                             use_idf=True,
                             ngram_range = (1,3))
train_vectors = vectorizer.fit_transform(Xtrain)

classifier_rbf = MultinomialNB(fit_prior=False)
classifier_rbf.fit(train_vectors, Ytrain)



def GetTweets(user_handle):
    tweetlist = api.user_timeline(screen_name=user_handle, count=20)
    allTweets = [tweet.text.encode("utf-8") for tweet in tweetlist]
    allTweets = pd.Series(allTweets)
    allTweets = GetCleanedData(allTweets)
    allTweets = allTweets.map(lambda x: ' '.join(x))
    allTweets = pd.Series(allTweets)
    prediction = classifier_rbf.predict(vectorizer.transform(allTweets))
    prediction = ["positive" if (x == 1) else "negative" for x in prediction]
    prediction = pd.Series(prediction)
    MyDataFrame = pd.DataFrame({'Tweets':allTweets,'sentiment':prediction})
    return MyDataFrame


#print GetTweets('harsha1007')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sentiment',methods=['POST'])
def sentiment():
    user=request.form['user']
    data=GetTweets(user)

    return render_template("sentiment.html", data=data.to_html())


if __name__ == '__main__':

    app.run(debug=True)
