import pandas as pd
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
# Tokenizer
from nltk.tokenize import word_tokenize
import requests
from collections import Counter
from itertools import chain
from time import time

# Part 3: Text mining.

# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with a string 
# corresponding to a path containing the wholesale_customers.csv file.
def read_csv_3(data_file):
	try:
		data = pd.read_csv(data_file, encoding='utf-8')
		return data
	except UnicodeDecodeError:
		try:
			# print('Unidecodecode error 1, encoding with latin1')
			data = pd.read_csv(data_file, encoding='latin1')
			return data
		except UnicodeDecodeError:
			# print('Unidecodecode error 2, encoding with ISO-8859-1')
			data = pd.read_csv(data_file, encoding='ISO-8859-1')
			return data

# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
	return df['Sentiment'].unique().tolist()

data = read_csv_3('./coronavirus_tweets.csv')
print(data)
sentiment_categories = get_sentiments(data)
print(sentiment_categories)

# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
	return data['Sentiment'].value_counts().index[1]

print("Second most popular sentiment:", second_most_popular_sentiment(data))

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
	dates_ex_pos = data[data['Sentiment'] == 'Extremely Positive']
	return dates_ex_pos['TweetAt'].value_counts().index[0]

print('Most common day for extremely positive tweets:', date_most_popular_tweets(data))

# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
	lower_case = df['OriginalTweet'].str.lower()
	df['OriginalTweet'] = lower_case
	return df

df = lower_case(data)
print(df)

# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.replace('[^a-zA-Z\s]', ' ', regex=True)
	return df

df = remove_non_alphabetic_chars(df)
print(df)

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
	# Make sure each word is separated by a single space
	df['OriginalTweet'] = df['OriginalTweet'].str.replace('\s+', ' ', regex=True)
	return df

df = remove_multiple_consecutive_whitespaces(df)
print(df)

# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
	tokenized_tweets = df['OriginalTweet'].tolist()
	df['OriginalTweet'] = [tweet.split() for tweet in tokenized_tweets]
	return df

start = time()
# This is not tokenising ? But might be a timesave
tdf = tokenize(df)
print("Time:", time() - start)
print(tdf)
print("")

# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
	word_count = sum(len(tweet) for tweet in tdf['OriginalTweet'])

	# text = tdf['OriginalTweet'].sum()
	print("number of words in all tweets:", word_count)
	return word_count

print("Counting words with repetitions...")
print(count_words_with_repetitions(tdf))

# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
    # Flatten the list of lists into a single list using chain
	all_words = list(chain.from_iterable(tdf['OriginalTweet']))
	unique_words = set(all_words)
	unique_word_count = len(unique_words)
	return unique_word_count

print(tdf)
print("Counting words without repetitions...")
unique_words = count_words_without_repetitions(tdf)
print(unique_words)

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf, k):
	all_words = [word.lower() for tweet in tdf['OriginalTweet'] for word in tweet]
	
	# Count the frequency of each word
	word_counts = Counter(all_words)

	# Find the most common words
	most_common_words = word_counts.most_common(k)
	return [word for word, count in most_common_words]

print("Most frequent words...")
most_frequent_words = frequent_words(tdf, 10)
print(most_frequent_words)

# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
	# URL from where to download the stop words
    stop_words_url = "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt"
    
    # Download the list of stop words
    response = requests.get(stop_words_url)
    # Ensure the request was successful

    response.raise_for_status()
	
    stop_words = set(response.content.decode().splitlines())
	
    # Function to filter out stop words and words with <=2 characters from a single tweet
    def filter_tweet(tweet_tokens):
        return [word for word in tweet_tokens if word.lower() not in stop_words and len(word) > 2]
	
    # Apply the filtering function to each tweet in the DataFrame
    # Assuming the column with tokenized tweets is named 'tokenized_tweet'; adjust if it's named differently
    # Apply the filtering function to each tweet in the DataFrame
    tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(filter_tweet)
    return tdf

print('Removing stop words...')
tdf = remove_stop_words(tdf)
print(tdf)

print(frequent_words(tdf, 10))

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
	# stemmer = PorterStemmer()
	stemmer = SnowballStemmer("english")
	# stemmer = LancasterStemmer()

	# tdf['OriginalTweet'] = [stemmer.stem(word) for word in tdf['OriginalTweet']]
	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda tweet: [stemmer.stem(word) for word in tweet])

	return tdf

print('Stemming...')
start = time()
tdf = stemming(tdf)
print("Time:", time() - start)
print(tdf)

# Repetition
print("Counting words with repetitions 2...")
print(count_words_with_repetitions(tdf))

print("Counting words without repetitions 2...")
print(count_words_without_repetitions(tdf))




from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
# Import count vectoriser
from sklearn.feature_extraction.text import CountVectorizer

# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 

def mnb_predict(df):	

	df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: ' '.join(x))

	X = df['OriginalTweet']
	y = df['Sentiment']

    # Encode the labels into numerical format
	le = LabelEncoder()
	y_encoded = le.fit_transform(y)

	# Pipeline that first vecorises the text then applies MultinomialNB Classifier
	model = make_pipeline(CountVectorizer(max_df = 2, min_df = 1, ngram_range=(4,4)), MultinomialNB())

	# Fit the model
	model = model.fit(X, y_encoded)
	predicted_sentiments = model.predict(X)

	# Convert numerical labels back to original text labels
	predicted_sentiments = le.inverse_transform(predicted_sentiments)

	return predicted_sentiments

print("\nPredicting Sentiments...")
start = time()
predicted_sentiments = mnb_predict(df)
print("Time:", time() - start)
print(predicted_sentiments)
print("Length of predicted sentiments:", len(predicted_sentiments))

# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
	correct = sum(y_pred == y_true)
	accuracy = correct / len(y_true)
	return round(accuracy, 4)

print("Accuracy:", mnb_accuracy(predicted_sentiments, df['Sentiment']))