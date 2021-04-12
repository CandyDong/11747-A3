#!/usr/bin/env python
import pickle
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import sys
from text_cnn_isarc import TextCNN
import os
from tensorflow.contrib import learn
import csv
from time import sleep
import pickle
import ast
from sklearn.metrics import f1_score
import copy
import math

#####################  GPU Configs  #################################

# Selecting the GPU to work on
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Desired graphics card config
session_conf = tf.ConfigProto(
	  allow_soft_placement=True,
	  log_device_placement=False,
	  gpu_options=tf.GPUOptions(allow_growth=True))

# Parameters
# ==================================================

np.random.seed(10)

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.5, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

print("loading data...")
x = pickle.load(open("./mainbalancedpickle.p","rb"))
revs, W, W2, word_idx_map, vocab, max_l = x[0], x[1], x[2], x[3], x[4], x[5]
print("data loaded!")# Load data


max_l = 100

# x_text_pos = []
# x_text_neg = []
x_text = []

# y_pos = []
# y_neg = []
y = []

test_x = []
test_y = []

pos, neg = 0, 0
for i in range(len(revs)):
	if revs[i]['split'] == "1":
		x_text.append(revs[i]['text'])
		y.append(ast.literal_eval(revs[i]['label']))
		if ast.literal_eval(revs[i]['label']) == [0, 1]: # sarcastic
			pos += 1
		elif ast.literal_eval(revs[i]['label']) == [1, 0]:
			neg += 1
		else:
			raise AssertionError("revs[{}]['label'] is {}".format(i, revs[i]['label']))
	elif revs[i]['split'] == "0":
		test_x.append(revs[i]['text'])
		test_y.append(ast.literal_eval(revs[i]['label']))
	else:
		raise AssertionError("split is {}".format(revs[i]['split']))

total = neg + pos

weight_for_0 = (1 / neg)*(total)/2.0 
weight_for_1 = (1 / pos)*(total)/2.0

class_weights = [[weight_for_0, weight_for_1]]

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

y = np.asarray(y)
test_y = np.asarray(test_y)


# get word indices
x = []
for i in range(len(x_text)):
	x.append(np.asarray([word_idx_map[word] for word in x_text[i].split()]))
# print("train word indices: {}".format(x))

x_test = []
for i in range(len(test_x)):
	x_test.append(np.asarray([word_idx_map[word] for word in test_x[i].split()]))

# padding
for i in range(len(x)):
	if( len(x[i]) < max_l ):
		x[i] = np.append(x[i],np.zeros(max_l-len(x[i])))		
	elif( len(x[i]) > max_l ):
		x[i] = x[i][0:max_l]
x = np.asarray(x)

for i in range(len(x_test)):
	if( len(x_test[i]) < max_l ):
		x_test[i] = np.append(x_test[i],np.zeros(max_l-len(x_test[i])))        
	elif( len(x_test[i]) > max_l ):
		x_test[i] = x_test[i][0:max_l]
x_test = np.asarray(x_test)
y_test = test_y

shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

# # balance training dataset
# pos_rows, pos_cols = np.where(y_train==[0,1]) # "sarcastic"
# pos_idx = [r for r, c in zip(pos_rows, pos_cols) if c == 0]
# total = len(y_train)
# pos = len(pos_idx)
# neg = total-pos

# num_copy = int(math.floor(neg/pos))-1
# print('Training examples:\n    Total: {}\n    \
# 	Positive: {} ({:.2f}% of total), \
# 	Negative: {} ({:.2f}% of total), \
# 	Num copy: {}\n'.format(
#     total, pos, 100 * pos / total, 
#     neg, 100 * neg / total, 
#     num_copy))
# x_train_pos, y_train_pos = np.take(x_train, pos_idx, axis=0), np.take(y_train, pos_idx, axis=0)
# x_train_pos_copies = np.repeat(x_train_pos, num_copy, axis=0)
# y_train_pos_copies = np.repeat(y_train_pos, num_copy, axis=0)

# x_train = np.vstack((x_train, x_train_pos_copies))
# y_train = np.vstack((y_train, y_train_pos_copies))
# assert(len(x_train) == len(y_train))

# shuffle_indices = np.random.permutation(np.arange(len(y_train)))
# x_train = x_train[shuffle_indices]
# y_train = y_train[shuffle_indices]

print("Train/Dev split: {:d}/{:d}, Test: {:d}".format(len(y_train), len(y_dev), len(y_test)))
x_train = np.asarray(x_train)
x_dev = np.asarray(x_dev)
y_train = np.asarray(y_train)
y_dev = np.asarray(y_dev)
word_idx_map["@"] = 0 # TODO: ?
rev_dict = {v: k for k, v in word_idx_map.items()}

# Weight for class 0: 0.60
# Weight for class 1: 2.93
# Train/Dev split: 2646/294, Test: 741


# Training
# ==================================================
with tf.Graph().as_default():

	sess = tf.Session(config=session_conf)
	with sess.as_default():
		cnn = TextCNN(
			sequence_length=max_l,
			num_classes=len(y_train[0]) ,
			vocab_size=len(vocab),
			word2vec_W = W,
			word_idx_map = word_idx_map,
			embedding_size=FLAGS.embedding_dim,
			batch_size=FLAGS.batch_size,
			filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
			num_filters=FLAGS.num_filters,
			class_weights=class_weights,
			l2_reg_lambda=FLAGS.l2_reg_lambda)

		# Define Training procedure
		global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer(1e-3)
		grads_and_vars = optimizer.compute_gradients(cnn.loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
		sess.run(tf.global_variables_initializer())


def train_step(x_batch, y_batch):
	"""
	A single training step
	"""
	feed_dict = {
	  cnn.input_x: x_batch,
	  cnn.input_y: y_batch,
	  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
	}
	_, step, loss, accuracy = sess.run(
		[train_op, global_step, cnn.loss, cnn.accuracy],
		feed_dict)
	time_str = datetime.datetime.now().isoformat()
	return loss, accuracy

def dev_step(x_batch, y_batch, writer=None):
	"""
	Evaluates model on a dev set
	"""
	feed_dict = {
	  cnn.input_x: x_batch,
	  cnn.input_y: y_batch,
	  cnn.dropout_keep_prob: 1.0
	}
	step, loss, conf_mat, y_pred, y_true = sess.run(
		[global_step, cnn.loss, cnn.confusion_matrix, cnn.predictions, cnn.correct_predictions],
		feed_dict)
	f_score = f1_score(y_true, y_pred, average='weighted')
	return loss, conf_mat, f_score
	

# Generate batches
batches = data_helpers.batch_iter(
	list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
dev_batches = data_helpers.batch_iter(
	list(zip(x_dev, y_dev)), FLAGS.batch_size, FLAGS.num_epochs)


train_loss = []
train_acc = []
best_acc = 0
for batch in batches:
	x_batch, y_batch = zip(*batch)
	x_batch = np.asarray(x_batch)
	y_batch = np.asarray(y_batch)
	t_loss, t_acc = train_step(x_batch, y_batch)
	current_step = tf.train.global_step(sess, global_step)
	train_loss.append(t_loss)
	train_acc.append(t_acc)
	if current_step % FLAGS.evaluate_every == 0:
		print("current step: {}".format(current_step))
		print("Train loss {:g}, Train acc {:g}".format(np.mean(np.asarray(train_loss)), np.mean(np.asarray(train_acc))))
		train_loss = []
		train_acc = []
		# Divide into batches
		dev_batches = data_helpers.batch_iter_dev(list(zip(x_dev, y_dev)), FLAGS.batch_size)
		dev_loss = []
		ll = len(dev_batches)
		conf_mat = np.zeros((2,2))
		for dev_batch in dev_batches:
			x_dev_batch = x_dev[dev_batch[0]:dev_batch[1]]
			y_dev_batch = y_dev[dev_batch[0]:dev_batch[1]]
			a, b, _ = dev_step(x_dev_batch, y_dev_batch)
			dev_loss.append(a)
			conf_mat += b
		valid_accuracy = float(conf_mat[0][0]+conf_mat[1][1])/len(y_dev)
		print("Valid loss {:g}, Valid acc {:g}".format(np.mean(np.asarray(dev_loss)), valid_accuracy))
		print("Valid - Confusion Matrix: ")
		print("conf_mat: {}".format(conf_mat))
		test_batches = data_helpers.batch_iter_dev(list(zip(x_test, y_test)), FLAGS.batch_size)
		test_loss = []
		test_f_score = []
		conf_mat = np.zeros((2,2))
		for test_batch in test_batches:
			x_test_batch = x_test[test_batch[0]:test_batch[1]]
			y_test_batch = y_test[test_batch[0]:test_batch[1]]
			a, b, c = dev_step(x_test_batch, y_test_batch)
			test_loss.append(a)
			test_f_score.append(c)
			conf_mat += b
		# print("Test loss {:g}, Test acc {:g}".format(
		# 	np.mean(np.asarray(test_loss)), 
		# 	float(conf_mat[0][0]+conf_mat[1][1])/len(y_test)))
		print("Test loss {:g}, Test acc {:g}, F score {:g}".format(
			np.mean(np.asarray(test_loss)), 
			float(conf_mat[0][0]+conf_mat[1][1])/len(y_test), 
			np.mean(np.asarray(test_f_score))))
		print("Test - Confusion Matrix: ")
		print("conf_mat: {}".format(conf_mat))
		sys.stdout.flush()
		if best_acc < valid_accuracy:
			best_acc = valid_accuracy
			directory = "./models"
			if not os.path.exists(directory):
				os.makedirs(directory)
			saver.save(sess, directory+'/main_balanced', global_step=1)






