import numpy as np
import tensorflow as tf
import sys
import os
import datetime
from sklearn.metrics import roc_auc_score 

import constants as const

for p in const.MODELS_PATHS:
	if os.path.exists(p):
		sys.path.append(p)

from nets.mobilenet import mobilenet_v2

for f in const.CHECKPOINT_FILE_PATHS:
	if os.path.exists(f):
		CHECKPOINT_FILE = f + '/mobilenet_v2_1.0_224.ckpt'

NUM_TUNING_RUNS = 100


def main():

	from data_loader import DataLoader
	from results_writer import ResultsWriter

	hyperparam_options = {
		'n_image_layers': [0, 1],
		'image_feature_sizes': np.arange(10, 100),
		'n_hidden_layers': [0, 1],
		'hidden_layer_sizes': np.arange(10, 300),
		'learning_rate': np.logspace(-2.5, -4.5),
		'activation_fn': [tf.nn.relu],#[tf.nn.sigmoid, tf.nn.relu, tf.nn.tanh],
		'dropout_pct': [0, .25, .5],
		'agg_method': ['concat', 'pool'],
		'train_mobilenet': [True, False],
		'mobilenet_endpoint': ['global_pool', 'Logits'],
		'max_epochs_no_improve': np.arange(3),
		'batch_size': [10],
		'dichotomize': [True]
	}

	resultcols = ['status']
	resultcols += ['fold']
	resultcols += [('mse_or_auc_%s' % o) for o in const.OUTCOMES]
	resultcols += list(hyperparam_options.keys())

	tuning_target = 'mse_or_auc_smoking'

	rw = ResultsWriter(resultcols)
	results_list = []

	for i in range(NUM_TUNING_RUNS):

		hyperparams = select_hyperparams(hyperparam_options)

		print('Running with the following hyperparams:')
		print(hyperparams)

		for val_fold in range(4):

			print('Training with val_fold =', val_fold)

			tf.reset_default_graph()
			dl = DataLoader(val_fold=val_fold)
			mdl = BaselineModel(dl, **hyperparams)

			fold_results = []

			try:

				with tf.compat.v1.Session() as s:

					train_stats, val_stats = mdl.train(s, **hyperparams)
					y_pred, y, loss_all = mdl.predict(s, 'val', hyperparams['batch_size'])

				if hyperparams['dichotomize']:

					mse_or_auc = [roc_auc_score(yt, yp) for yt, yp in zip(y.T, y_pred.T)]

				else:

					mse_or_auc = np.mean((y - y_pred) ** 2, axis=0)

				mse_or_auc_dict = {('mse_or_auc_%s' % o): v for o, v in zip(
					const.OUTCOMES, mse_or_auc)}

				fold_results.append(mse_or_auc_dict)

				rw.write(i, {
					'status': 'complete',
					'fold': val_fold,
					**mse_or_auc_dict,
					**hyperparams})
				rw.plot(
					i, train_stats, val_stats, y_pred, y, const.OUTCOMES, mse_or_auc,
					**hyperparams)

			except:

				rw.write(i, {'status': 'failed', **hyperparams})

			if len(fold_results) > 0:

				result = np.mean([x[tuning_target] for x in fold_results])
				results_list.append((hyperparams, result))

	hps, results = list(zip(*results_list))
	hyperparams = hps[np.argmax(results)]

	tf.reset_default_graph()
	dl = DataLoader(val_fold=3)
	mdl = BaselineModel(dl, **hyperparams)

	try:

		with tf.compat.v1.Session() as s:

			train_stats, val_stats = mdl.train(s, **hyperparams)
			y_pred, y, loss_all = mdl.predict(s, 'test', hyperparams['batch_size'])

		if hyperparams['dichotomize']:

			mse_or_auc = [roc_auc_score(yt, yp) for yt, yp in zip(y.T, y_pred.T)]

		else:

			mse_or_auc = np.mean((y - y_pred) ** 2, axis=0)

		mse_or_auc_dict = {('mse_or_auc_%s' % o): v for o, v in zip(
			const.OUTCOMES, mse_or_auc)}

		rw.write('final', {
			'status': 'complete',
			'fold': 4,
			**mse_or_auc_dict,
			**hyperparams})
		rw.plot(
			'final', train_stats, val_stats, y_pred, y, const.OUTCOMES, mse_or_auc,
			**hyperparams)

	except:

		rw.write('final', {'status': 'failed', **hyperparams})


class BaselineModel:

	def __init__(
		self, dataloader,
		n_image_layers=1,
		image_feature_sizes=50,
		n_hidden_layers=1,
		hidden_layer_sizes=50,
		learning_rate=1e-3,
		activation_fn=tf.nn.relu,
		dropout_pct=.5,
		train_mobilenet=False,
		agg_method='pool',
		mobilenet_endpoint='global_pool',
		**kwargs):

		self.dataloader = dataloader

		self.n_out = dataloader.n_out
		self.n_images = dataloader.n_images

		self.image_feature_sizes = [image_feature_sizes] * n_image_layers
		self.hidden_layer_sizes = [hidden_layer_sizes] * n_hidden_layers

		self.learning_rate = learning_rate
		self.activation_fn = activation_fn
		self.dropout_pct = dropout_pct

		self.train_mobilenet = train_mobilenet

		self.agg_method = agg_method
		self.mobilenet_endpoint = mobilenet_endpoint

		self._build_placeholders()
		self._build_mobilenet()
		self._build_model()
		self._build_train_step()


	def train(
		self, sess,
		max_epochs=30, max_epochs_no_improve=2,
		batch_size=20, batch_eval_freq=1,
		verbose=True,
		**kwargs):

		sess.run(tf.compat.v1.global_variables_initializer())
		self.mobilenet_saver.restore(sess, CHECKPOINT_FILE)

		batches_per_epoch = int(np.ceil(
			self.dataloader.n_train / batch_size))

		xval, yval = self.dataloader.sample_data(part='val', n=-1)

		train_stats = []
		val_stats = []

		val_loss = []

		for batch_idx, (xb, yb) in enumerate(self.dataloader.get_batch(
			'val', batch_size)):

			print('Starting val batch %i' % batch_idx)

			loss_ = sess.run(
				self.loss,
				feed_dict={self.x: xb, self.y: yb, self.is_training: False})

			val_loss.append((len(xb), loss_))

		val_loss = sum([l * v for l, v in val_loss]) / sum([l for l, v in val_loss])

		val_stats.append((0, val_loss))

		if verbose:

			print('Initial val loss: %.2f' % val_stats[-1][1])

		best_val_loss = val_stats[0][1]
		n_epochs_no_improve = 0

		for epoch_idx in range(max_epochs):

			for batch_idx, (xb, yb) in enumerate(self.dataloader.get_batch(
				'train', batch_size)):

				print('Starting training batch %i' % batch_idx)

				loss_, _ = sess.run(
					[self.loss, self.train_step],
					feed_dict={self.x: xb, self.y: yb, self.is_training: True})

				if batch_idx % batch_eval_freq == 0:
					idx = epoch_idx * batches_per_epoch + batch_idx
					train_stats.append((idx, loss_))

			idx = (epoch_idx + 1) * batches_per_epoch

			val_loss = []

			for batch_idx, (xb, yb) in enumerate(self.dataloader.get_batch(
				'val', batch_size)):

				print('Starting val batch %i' % batch_idx)

				loss_ = sess.run(
					self.loss,
					feed_dict={self.x: xb, self.y: yb, self.is_training: False})

				val_loss.append((len(xb), loss_))

			val_loss = sum([l * v for l, v in val_loss]) / sum([l for l, v in val_loss])

			val_stats.append((idx, val_loss))

			print('Completed Epoch %i' % epoch_idx)

			if verbose:

				print('Val loss: %.2f, Train loss: %.2f' % (
					val_stats[-1][1],
					np.mean(list(zip(*train_stats[-batches_per_epoch:]))[1])
				))

			if val_stats[-1][1] < best_val_loss:
				best_val_loss = val_stats[-1][1]
				n_epochs_no_improve = 0
			else:
				n_epochs_no_improve += 1

			if n_epochs_no_improve > max_epochs_no_improve:
				break

		return train_stats, val_stats


	def predict(self, sess, part, batch_size):

		assert part in ['all', 'train', 'val', 'test']

		y_pred = []
		y = []
		loss = []

		for batch_idx, (xb, yb) in enumerate(self.dataloader.get_batch(
			part, batch_size)):

			if self.dataloader.dichotomize:

				y_pred_, loss_ = sess.run(
					[self.y_prob_pred, self.loss],
					feed_dict={self.x: xb, self.y: yb, self.is_training: False})

			else:

				y_pred_, loss_ = sess.run(
					[self.y_pred, self.loss],
					feed_dict={self.x: xb, self.y: yb, self.is_training: False})

			y_pred.append(y_pred_)
			y.append(yb)
			loss.append((len(xb), loss_))

		loss = sum([l * v for l, v in loss]) / sum([l for l, v in loss])
		y_pred = np.concatenate(y_pred, axis=0)
		y = np.concatenate(y, axis=0)

		return y_pred, y, loss


	def _build_placeholders(self):

		self.x = tf.placeholder(
			dtype=tf.float32,
			shape=(None, self.n_images, 224, 224, 3))

		self.y = tf.placeholder(
			dtype=tf.float32,
			shape=(None, self.n_out))

		self.is_training = tf.placeholder(
			dtype=tf.bool,
			shape=())


	def _build_mobilenet(self):

		x_flat = tf.reshape(self.x, (-1, 224, 224, 3))

		if self.train_mobilenet:
			is_training = self.is_training
		else:
			is_training = False

		with tf.contrib.slim.arg_scope(
			mobilenet_v2.training_scope(is_training=is_training)):
			
			logits, endpoints = mobilenet_v2.mobilenet(x_flat)

		ema = tf.train.ExponentialMovingAverage(0.999)
		self.mobilenet_saver = tf.compat.v1.train.Saver(
			ema.variables_to_restore())

		features_flat = endpoints[self.mobilenet_endpoint]

		if self.mobilenet_endpoint == 'global_pool':
			features_flat = tf.squeeze(features_flat, [1, 2])

		self.image_features = tf.reshape(
			features_flat,
			(-1, self.n_images, const.MOBILENET_OUTPUT_SIZE[self.mobilenet_endpoint]))


	def _build_model(self):

		with tf.compat.v1.variable_scope('image_features'):

			feat = mlp(
				self.image_features,
				self.image_feature_sizes,
				dropout_pct=self.dropout_pct,
				activation_fn=self.activation_fn,
				training=self.is_training)

		feat = self.image_features

		if self.agg_method == 'pool':

			feature_vec = tf.concat(
				[tf.reduce_mean(feat, axis=1), tf.reduce_max(feat, axis=1)],
				axis=1)

		elif self.agg_method == 'concat':

			image_feature_layers = [const.MOBILENET_OUTPUT_SIZE[self.mobilenet_endpoint]]
			image_feature_layers += self.image_feature_sizes

			feature_vec = tf.reshape(
				feat,
				(-1, image_feature_layers[-1] * self.n_images))

		with tf.compat.v1.variable_scope('outcomes'):

			with tf.compat.v1.variable_scope('mlp'):

				hidden_layer = mlp(
					feature_vec,
					self.hidden_layer_sizes,
					dropout_pct=self.dropout_pct,
					activation_fn=self.activation_fn,
					training=self.is_training)

			with tf.compat.v1.variable_scope('linear'):

				self.y_pred = mlp(
					hidden_layer,
					[self.n_out],
					dropout_pct=self.dropout_pct,
					activation_fn=None,
					training=self.is_training)

			if self.dataloader.dichotomize:

				self.y_prob_pred = tf.nn.sigmoid(self.y_pred)


	def _build_train_step(self):

		if self.dataloader.dichotomize:

			self.loss = tf.reduce_mean(
				tf.nn.sigmoid_cross_entropy_with_logits(
					labels=self.y,
					logits=self.y_pred))

		else:

			self.loss = tf.compat.v1.losses.mean_squared_error(
				self.y,
				self.y_pred)

		if self.train_mobilenet:

			self.train_step = tf.compat.v1.train.AdamOptimizer(
				self.learning_rate).minimize(self.loss)

		else:

			myvars = tf.compat.v1.get_collection(
				tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
				scope='image_features')

			myvars += tf.compat.v1.get_collection(
				tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
				scope='outcomes')

			self.train_step = tf.compat.v1.train.AdamOptimizer(
				self.learning_rate).minimize(self.loss, var_list=myvars)


def mlp(x, hidden_layer_sizes,
		dropout_pct=0.,
		activation_fn=tf.nn.relu,
		training=True,
		reuse=False):

	hidden_layer = x

	with tf.compat.v1.variable_scope('mlp', reuse=reuse):

		for i, layer_size in enumerate(hidden_layer_sizes):

			hidden_layer = tf.layers.dense(
				hidden_layer, layer_size,
				activation=activation_fn,
				name='fc_%i' % i,
				reuse=reuse)

			if dropout_pct > 0:
				hidden_layer = tf.layers.dropout(
					hidden_layer, rate=dropout_pct,
					training=training,
					name='dropout_%i' % i)

	return hidden_layer


def select_hyperparams(hpdict):

	return {k: np.random.choice(v) for k, v in hpdict.items()}


if __name__ == '__main__':
	main()
