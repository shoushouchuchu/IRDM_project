import numpy as np
import csv
import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn import svm
from sklearn import metrics
import math
import random
import datetime
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import sys
import time
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
import datetime
from scipy import sparse, io
import string
import nltk
from nltk.tokenize import TweetTokenizer
from scipy.sparse import coo_matrix, vstack
from datetime import datetime
from sklearn.grid_search import GridSearchCV

def unfussy_reader(csv_reader):
    while True:
        try:
            yield next(csv_reader)
        except csv.Error:
            print("Problem with some row")
            continue

tweet_ids_dict = {}



################ read and prepare datasets #############################################
with open('Chu_geofilter_tweets_doc2vec_2016.txt', 'rb') as tsvin_Chu_file:
	tsvin_Chu = unfussy_reader(csv.reader(tsvin_Chu_file, delimiter = '\t'))
	counter = 0
	row_counter = 0
	for row in tsvin_Chu:
		row_counter += 1
		if len(row) != 13:
			print "Problem with row", counter
			print "now it is row ", row_counter
			counter += 1
			continue
		date = row[0]
		time = row[1]
		tweet_id = row[2]
		user_id = row[3]
		retweet = int(row[4])
		retweet_including_tweets_starting_with_RT = int(row[5])
		user_location = row[6]
		tweet_latitude = float(row[7])
		tweet_longitude = float(row[8])
		coordinate_is_in_england = int(row[9])
		platform = row[10]
		twokens = row[11]
		raw_tweet_text = row[12]
		if coordinate_is_in_england == 1:
			tweet_ids_dict[tweet_id] = [date, twokens, raw_tweet_text]

with open('Chu_result_dict.json', 'w') as fp:
	json.dump(tweet_ids_dict, fp)



with open('Chu_result_dict.json', 'r') as f_in:
    tweet_ids_dict = json.load(f_in)



tweet_id_raw_tweet_dict = {}
tweet_id_tokens_dict = {}
tweet_id_date_dict = {}

for tweet_id, item in tweet_ids_dict.iteritems():
	tweet_id_date_dict[tweet_id] = item[0]
	tweet_id_tokens_dict[tweet_id] = item[1]
	tweet_id_raw_tweet_dict[tweet_id] = item[2]



with open('tweet_id_date_dict.json', 'w') as f_date:
	json.dump(tweet_id_date_dict, f_date)


with open('tweet_id_tokens_dict.json', 'w') as f_tokens:
	json.dump(tweet_id_tokens_dict, f_tokens)

with open('tweet_id_raw_tweet_dict.json', 'w') as f_raw_tweet:
	json.dump(tweet_id_raw_tweet_dict, f_raw_tweet)

with open('tweet_id_tokens_dict.json', 'r') as f_tokens:
	tweet_id_tokens_dict = json.load(f_tokens)



#######################Get words by training Twitter-based word embedding ################################

# import modules & set up logging
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = tweet_id_raw_tweet_dict.values()
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count = 10, size = 256, workers = 4, window = 5)
model.save(fname)
model = Word2Vec.load(fname)

# example search
model.most_similar(positive=['woman', 'king'], negative=['man'])


####################  Get leave EU tweet ids    ############################################################
leave_EU_more_than_unigrams = [['vote', 'leave'], ['leave', 'eu'], ['exit', 'eu'], ['leave', 'europe'], ['exit', 'europe'], ['vote', 'exit']]
leave_EU_words_unigrams = ['#Brexit', 'Brexit', '#Lexit', 'Lexit', '#EUreferendum', 'EUreferendum', '#Lexit', 'Lexit', '@vote_leave',  '#EUref', 'EUref', '@LeaveEUOfficial', '@BetterOffOut', 'anti-EU', '@MrBrexit', 'MrBrexit', '#EURef', 'EURef', '@voteleave', 'Referendum', '#VoteLeave', '@euromove', 'euromove', '@VoteLeaveUKIP', '@ExittheEU', '#LeaveEU', '#voteleavetakecontrol', '@OutandProudUK', '@Stop_The_EU', '@BeLeaveBritain', '@VoteLeave_eu', '@NoThanksEU', '@Vote_LeaveMedia', '@VoteLeaveCymru']
lowercased_leave_EU_words_unigrams = [x.lower() for x in leave_EU_words_unigrams]

leave_EU_tweet_ids = []
set_leave_EU_words_unigrams = set(lowercased_leave_EU_words_unigrams)
length_leave_bigrams = len(leave_EU_more_than_unigrams)

counter = 0
for key, item in tweet_id_tokens_dict.iteritems():
	counter += 1
	print counter
	item_lower_word_list = item.lower().split()
	intersection_set_unigram = set(item_lower_word_list) & set_leave_EU_words_unigrams
	if not bool(intersection_set_unigram):
		flag = False
		acc = 0
		while ((not flag) and acc < length_leave_bigrams):
			flag = (leave_EU_more_than_unigrams[acc][0] in item_lower_word_list) and (leave_EU_more_than_unigrams[acc][1] in item_lower_word_list)
			acc += 1
		if flag:
			leave_EU_tweet_ids.append(key)
	else:
		leave_EU_tweet_ids.append(key)

with open('selected_leave_EU_tweet_ids', 'w') as file_leave_EU:
	pickle.dump(leave_EU_tweet_ids, file_leave_EU)


###############Get stay EU tweet ids #############################################
stay_EU_more_than_unigrams = [['vote', 'stay'], ['stay', 'eu'], ['stay', 'europe'], ['remain', 'EU'], ['remain', 'europe']]
stay_EU_words_unigrams = ['@stayinEU', '@remaininEU', '@uktostayeu', '@sayyes2europe', '@StrongerIn', '#EURef', 'EURef', '@votestay', 'pro-EU', '@ConsForstay', '@BestayBritain', '#StrongerIN', 'VoteStay']

lowercased_stay_EU_words_unigrams = [x.lower() for x in stay_EU_words_unigrams]

stay_EU_tweet_ids = []
set_stay_EU_words_unigrams = set(lowercased_stay_EU_words_unigrams)
length_stay_bigrams = len(stay_EU_more_than_unigrams)

counter = 0
for key, item in tweet_id_tokens_dict.iteritems():
	counter += 1
	print counter
	item_lower_word_list = item.lower().split()
	intersection_set_unigram = set(item_lower_word_list) & set_stay_EU_words_unigrams
	if not bool(intersection_set_unigram):
		flag = False
		acc = 0
		while ((not flag) and acc < length_stay_bigrams):
			flag = (stay_EU_more_than_unigrams[acc][0] in item_lower_word_list) and (stay_EU_more_than_unigrams[acc][1] in item_lower_word_list)
			acc += 1
		if flag:
			stay_EU_tweet_ids.append(key)
	else:
		stay_EU_tweet_ids.append(key)

with open('selected_stay_EU_tweet_ids', 'w') as file_stay_EU:
	pickle.dump(stay_EU_tweet_ids, file_stay_EU)


with open('tweet_id_raw_tweet_dict.json', 'r') as f_raw_tweet:
	tweet_id_raw_tweet_dict = json.load(f_raw_tweet)

file_leave_EU = open('selected_leave_EU_tweet_ids', 'r')
selected_leave_EU_tweet_ids = pickle.load(file_leave_EU)

file_stay_EU = open('selected_stay_EU_tweet_ids', 'r')
selected_stay_EU_tweet_ids = pickle.load(file_stay_EU)


############################## get raw tweets given the tweet ids########################################################

leave_EU_tweet_id_raw_text_dict = {}
stay_EU_tweet_id_raw_text_dict = {}

for tweet_id in selected_leave_EU_tweet_ids:
	leave_EU_tweet_id_raw_text_dict[tweet_id] = tweet_id_raw_tweet_dict.get(tweet_id)

for tweet_id in selected_stay_EU_tweet_ids:
	stay_EU_tweet_id_raw_text_dict[tweet_id] = tweet_id_raw_tweet_dict.get(tweet_id)

with open('leave_EU_tweet_id_raw_text_dict.pickle', 'wb') as f_raw_text_leave_EU_dict:
	pickle.dump(leave_EU_tweet_id_raw_text_dict, f_raw_text_leave_EU_dict)

with open('stay_EU_tweet_id_raw_text_dict.pickle', 'wb') as f_raw_text_stay_EU_dict:
	pickle.dump(stay_EU_tweet_id_raw_text_dict, f_raw_text_stay_EU_dict)

# ###################filter the intersection twets of stay and leave EU #################################################

# with open('leave_EU_tweet_id_raw_text_dict.pickle', 'rb') as f_raw_text_leave_EU_dict:
# 	leave_EU_tweet_id_raw_text_dict = pickle.load(f_raw_text_leave_EU_dict)

# with open('stay_EU_tweet_id_raw_text_dict.pickle', 'rb') as f_raw_text_stay_EU_dict:
# 	stay_EU_tweet_id_raw_text_dict = pickle.load(f_raw_text_stay_EU_dict)

# intersection_leave_stay = list(set(stay_EU_tweet_id_raw_text_dict.keys()) & set(leave_EU_tweet_id_raw_text_dict.keys()))

# filtered_leave_tweets = {}
# filtered_stay_tweets = {}

# for leave_key, leave_value in leave_EU_tweet_id_raw_text_dict.iteritems():
# 	if leave_key not in intersection_leave_stay:
# 		filtered_leave_tweets[leave_key] = leave_value

# for stay_key, stay_value in stay_EU_tweet_id_raw_text_dict.iteritems():
# 	if stay_key not in intersection_leave_stay:
# 		filtered_stay_tweets[stay_key] = stay_value

# with open('filtered_leave_tweets.pickle', 'wb') as f_filtered_leave_tweets:
# 	pickle.dump(filtered_leave_tweets, f_filtered_leave_tweets)

# with open('filtered_stay_tweets.pickle', 'wb') as f_filtered_stay_tweets:
# 	pickle.dump(filtered_stay_tweets, f_filtered_stay_tweets)



with open('filtered_leave_tweets.pickle', 'rb') as f_filtered_leave_tweets:
	filtered_leave_tweets = pickle.load(f_filtered_leave_tweets)

with open('filtered_stay_tweets.pickle', 'rb') as f_filtered_stay_tweets:
	filtered_stay_tweets = pickle.load(f_filtered_stay_tweets)

stay_tweets_5000 = filtered_stay_tweets.values()[0:5000]
leave_tweets_5000 = filtered_leave_tweets.values()[0:5000]

stay_labels_5000 = [1] * 5000
leave_labels_5000 = [0] * 5000

fold_number = 10
stay_leave_tweets = []
stay_leave_labels = []

for j in range(fold_number):
	stay_leave_tweets += stay_tweets_5000[500*j: 500 + 500*j] + leave_tweets_5000[500*j: 500 + 500*j]
	stay_leave_labels += stay_labels_5000[500*j: 500 + 500*j] + leave_labels_5000[500*j: 500 + 500*j]


def tokenize(text):
	tknzr = TweetTokenizer()
    tokens = tknzr.tokenize(text)
    tokens_remove_punctuation = [i for i in tokens if i not in string.punctuation]
    return tokens_remove_punctuation


################################## vectorise tweets and get features n-grams (1 <= n <= 3)########################
vect_unigram = CountVectorizer(ngram_range = (1, 1), min_df = 2, stop_words = 'english', tokenizer=tokenize)
X_unigram = vect_unigram.fit_transform(stay_leave_tweets)

vect_bigram = CountVectorizer(ngram_range = (2, 2), min_df = 2, binary = True, tokenizer=tokenize)
X_bigram = vect_bigram.fit_transform(stay_leave_tweets)

vect_trigram = CountVectorizer(ngram_range = (3, 3), min_df = 2, binary = True, tokenizer=tokenize)
X_trigram = vect_trigram.fit_transform(stay_leave_tweets)

X = csr_matrix(hstack((X_unigram, X_bigram, X_trigram)))

y = np.array(stay_leave_labels)

C_range = np.array([1, 2, 5, 10, 15, 20, 50, 100])
gamma_range = np.array([0.2, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])

label_length = len(stay_leave_labels)
each_fold_number = int(label_length / fold_number)
full_list = range(0, label_length)

for ind in range(0, fold_number):
	
	print "Now it is fold %d" % (ind + 1)
	print datetime.now()
	
	starting = ind * each_fold_number
	ending = (ind + 1) * each_fold_number

	test_index = range(starting, ending)
	train_index = list(set(full_list) - set(test_index))
	
	X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]

	param_grid_svm_rbf = dict(gamma = gamma_range, C = C_range)
	grid_svm_rbf = GridSearchCV(estimator = svm.SVC(kernel = 'rbf'), param_grid = param_grid_svm_rbf)
	grid_svm_rbf.fit(X_train, y_train)
	print("The best parameters for SVM with rbf model are %s with a score of %0.2f" % (grid_svm_rbf.best_params_, grid_svm_rbf.best_score_))
	optimal_C = grid_svm_rbf.best_params_.get('C')
	optimal_gamma = grid_svm_rbf.best_params_.get('gamma')

	clf_svm_rbf = svm.SVC(kernel = 'rbf', C = optimal_C, gamma = optimal_gamma)
	predicted_svm_rbf = clf_svm_rbf.fit(X_train, y_train).predict(X_test)
	print "SVM with rbf model report for fold using fold %d as testing dataset is " % (ind + 1)
	print metrics.classification_report(y_test, predicted_svm_rbf)
	print confusion_matrix(y_test, predicted_svm_rbf)

	print datetime.now()

alphas_range = 10. ** np.arange(-5, 4)

fold_number = 5
each_fold_number = 2000

for ind in range(0, fold_number):
	
	print "Now it is fold %d" % (ind + 1)
	print datetime.now()
	
	starting = ind * each_fold_number
	ending = (ind + 1) * each_fold_number

	test_index = range(starting, ending)
	train_index = list(set(full_list) - set(test_index))
	
	X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]

	param_grid_ridge = dict(alpha = alphas_range)
	grid_ridge = GridSearchCV(estimator = RidgeClassifier(), param_grid = param_grid_ridge, cv = 5)
	grid_ridge.fit(X_train, y_train)
	print("The best parameters for ridge classifier is %s with a score of %0.2f" % (grid_ridge.best_params_, grid_ridge.best_score_))
	optimal_alpha = grid_ridge.best_params_.get('alpha')

	clf_ridge = RidgeClassifier(alpha = optimal_alpha)
	predicted_ridge = clf_ridge.fit(X_train, y_train).predict(X_test)
	print "Ridge classifier model report for fold using fold %d as testing dataset is " % (ind + 1)
	print metrics.classification_report(y_test, predicted_ridge)

	print datetime.now()


