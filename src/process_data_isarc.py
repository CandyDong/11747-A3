#!/usr/bin/env python
import pandas as pd
import sys, re
import numpy as np
import pickle
from collections import defaultdict
import json
import tweepy
import os
import csv

consumer_key = 'wr8aGHPssCGVrPmAi1hqcGrhf'
consumer_key_secret = 'HwwwI3jKjS7xxw1ew7xEt4G5MxnDdwzfn7fWgGSp4OzvX0vEE8'
access_token = '1244812419165294592-ph7HNoPbeJuJOAuQZJblVpiOv0pf0w'
access_token_secret = 'kYAgukPuvZTROU5x0bANnNC1WyrufPAiDAzn232gXm5So'

auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

COMMENTS_FILE = "../data/comments.json"
TRAIN_MAP_FILE = "../data/isarcasm_train.csv"
TEST_MAP_FILE = "../data/isarcasm_test.csv"

# saves the actual data on the fly since tweet has data fetch limit
TRAIN_DATASET_FILE = "../data/isarcasm_train_data.csv"
TEST_DATASET_FILE = "../data/isarcasm_test_data.csv"

DATUM_KEYS = sorted(["i", "y", "id", "text", "label", "num_words", "split"])

def load_csv(dataset_file):
	revs = []
	if not os.path.exists(dataset_file):
		return revs

	i = 0
	with open(dataset_file, 'r') as f:
		reader = csv.DictReader(f)
		for row in reader:
			i = int(row["i"]) + 1

			row.pop("i")
			revs.append(row)
			
	return revs, i

def fetch_tweet(tweet_id):
	tweetFetched = api.get_status(tweet_id)
	return tweetFetched.text

def construct_vocab(revs, vocab):
	for rev in revs:
		words = set(orig_rev.split())
		for word in words:
			vocab[word] += 1

def fetch_data(data, data_type, clean_string=True):
	dataset_file = None
	if data_type == "train":
		dataset_file = TRAIN_DATASET_FILE
	elif data_type == "test":
		dataset_file = TEST_DATASET_FILE
	else:
		raise AssertionError("data_type is {}".format(data_type))

	revs, start = load_csv(dataset_file)
	print("{} {} data loaded.....".format(len(revs), data_type))

	if start == len(data):
		print("all {} data loaded.....".format(data_type))
		return	

	with open(dataset_file, 'w') as f:
		writer = csv.DictWriter(f, fieldnames=DATUM_KEYS)
		writer.writeheader()

		for i, line in enumerate(data[start:]):
			rev = []
			label_str = line[1]
			if (label_str == "not_sarcastic"):
				label = [1, 0]
			elif (label_str =="sarcastic"):
				label = [0, 1]
			else:
				raise AssertionError("label_str is {}".format(label_str))

			tweet_id = line[0]
			try:
				tweet = fetch_tweet(tweet_id)
				# print("tweet: {}".format(tweet))
			except tweepy.error.TweepError as e:
				# reached rate limit
				if e.args[0][0]['code'] == 88:
					raise AssertionError("Rate limit exceeded at {} {}. Run the script again later.".format(data_type, i))
				# print("{}, id={}, i={}".format(e, tweet_id, i))
				continue

			rev.append(tweet.strip())
			if clean_string:
				orig_rev = clean_str(" ".join(rev))
			else:
				orig_rev = " ".join(rev).lower()
			orig_rev = (orig_rev.split())[0:100]
			orig_rev = " ".join(orig_rev)

			datum = {"y": int(1), # TODO: figure out what this is
					 "id": tweet_id,
					 "text": orig_rev,
					 "label": label,
					 "num_words": len(orig_rev.split()),
					 "split": int(1) if data_type == "train" else int(0)}
			revs.append(datum)

			# write collected data to csv file
			datum_plus = {"i": i}
			datum_plus.update(datum)
			writer.writerow(datum_plus)

			if i % 10 == 0:
				print("{} {} data scanned.....".format(i, data_type))

	return revs


def build_data_cv(data_folder, cv=10, clean_string=True):
	isarc_train_file = data_folder[0]
	isarc_test_file = data_folder[1]

	train_data = np.asarray(pd.read_csv(isarc_train_file))
	test_data = np.asarray(pd.read_csv(isarc_test_file))
	# print("train_data: {}".format(train_data))
	# tweet_id | sarcasm_label | sarcasm_type


	revs = []
	revs_train = fetch_data(train_data, "train", clean_string=True)
	revs_test = fetch_data(test_data, "test", clean_string=True)
	print("loaded train dataset size = {}, test dataset size ={}".format(len(revs_train), len(revs_test)))

	revs.append(revs_train)
	revs.append(revs_test)

	vocab = defaultdict(float)
	construct_vocab(revs)

	return revs, vocab


def clean_str(string, TREC=False):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Every dataset is lower cased except for TREC
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
	string = re.sub(r"\'s", " \'s", string) 
	string = re.sub(r"\'ve", " \'ve", string) 
	string = re.sub(r"n\'t", " n\'t", string) 
	string = re.sub(r"\'re", " \'re", string) 
	string = re.sub(r"\'d", " \'d", string) 
	string = re.sub(r"\'ll", " \'ll", string) 
	string = re.sub(r",", " , ", string) 
	string = re.sub(r"!", " ! ", string) 
	string = re.sub(r"\(", " \( ", string) 
	string = re.sub(r"\)", " \) ", string) 
	string = re.sub(r"\?", " \? ", string) 
	string = re.sub(r"\s{2,}", " ", string)    
	return string.strip() if TREC else string.strip().lower()


if __name__=="__main__":  

	data_folder = [TRAIN_MAP_FILE,TEST_MAP_FILE] 
	print("loading data...")
	revs, vocab = build_data_cv(data_folder,  cv=10, clean_string=True)
	max_l = np.max(pd.DataFrame(revs)["num_words"])

	print("number of sentences: " + str(len(revs)))
	print("vocab size: " + str(len(vocab)))
	print("max sentence length: " + str(max_l))

	print("loading word2vec vectors...")
	# w2v_file = sys.argv[1] # fasttext embedding file



