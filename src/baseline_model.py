import numpy as np
import tensorflow as tf
import sys
import os
import datetime

MODELS_PATHS = [
	'/Users/mme/projects/models/research/slim',
	'/scratch/mme4/models/research/slim'
]

for p in MODELS_PATHS:
	if os.path.exists(p):
		sys.path.append(p)

from nets.mobilenet import mobilenet_v2

CHECKPOINT_FILE_PATHS = [
	'/Users/mme/projects/imageturk/mobilenet_checkpoint',
	'/scratch/mme4/mobilenet_checkpoint'
]

for f in CHECKPOINT_FILE_PATHS:
	if os.path.exists(f):
		CHECKPOINT_FILE = f + '/mobilenet_v2_1.0_224.ckpt'

GLOBAL_POOL_N_FEATURES = 1280

import constants as const


def main():

	from data_loader import DataLoader
	from results_writer import ResultsWriter

	hyperparam_options = {
		'image_feature_size': np.arange(10, 300),
		'n_hidden_layers': [0, 1, 1, 1],
		'hidden_layer_sizes': np.arange(10, 300),
		'learning_rate': np.exp(np.linspace(-3, -10, 10)),
		'activation_fn': [tf.nn.sigmoid, tf.nn.relu, tf.nn.tanh],
		'dropout_pct': [0, .25, .5],
		'train_mobilenet': [True, False],
		'max_epochs_no_improve': np.arange(3),
		'batch_size': [10],
		'val_fold': np.arange(4)
	}

	resultcols = ['status']
	resultcols += [('mse_%s' % o) for o in const.OUTCOMES]
	resultcols += list(hyperparam_options.keys())

	rw = ResultsWriter(resultcols)

	for i in range(30):

		tf.reset_default_graph()

		hyperparams = select_hyperparams(hyperparam_options)

		dl = DataLoader(**hyperparams)

		mdl = BaselineModel(dl, **hyperparams)

		#try:

		with tf.compat.v1.Session() as s:

			train_stats, val_stats = mdl.train(s, max_epochs=1, **hyperparams)
			y_pred, y, mse_all = mdl.predict(s, 'val', hyperparams['batch_size'])

		mse = np.mean((y - y_pred) ** 2, axis=0)
		mse_dict = {('mse_%s' % o): v for o, v in zip(const.OUTCOMES, mse)}

		rw.write(i, {'status': 'complete', **mse_dict, **hyperparams})
		rw.plot(i, train_stats, val_stats, y_pred, y, const.OUTCOMES, mse)

		#except:

		#	rw.write(i, {'status': 'failed', **hyperparams})


class BaselineModel:

	def __init__(
		self, dataloader,
		image_feature_size=50,
		n_hidden_layers=1,
		hidden_layer_sizes=50,
		learning_rate=1e-3,
		activation_fn=tf.nn.relu,
		dropout_pct=.5,
		train_mobilenet=False,
		**kwargs):

		self.dataloader = dataloader

		self.n_out = dataloader.n_out
		self.n_images = dataloader.n_images

		self.image_feature_size = image_feature_size
		self.hidden_layer_sizes = [hidden_layer_sizes] * n_hidden_layers

		self.learning_rate = learning_rate
		self.activation_fn = activation_fn
		self.dropout_pct = dropout_pct

		self.train_mobilenet = train_mobilenet

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

		val_mse = []

		for batch_idx, (xb, yb) in enumerate(self.dataloader.get_batch(
			'val', batch_size)):

			print('Starting val batch %i' % batch_idx)

			loss_ = sess.run(
				self.loss,
				feed_dict={self.x: xb, self.y: yb, self.is_training: False})

			val_mse.append((len(xb), loss_))

		val_mse = sum([l * mse for l, mse in val_mse]) / sum([l for l, mse in val_mse])

		val_stats.append((0, val_mse))

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

			val_mse = []

			for batch_idx, (xb, yb) in enumerate(self.dataloader.get_batch(
				'val', batch_size)):

				print('Starting val batch %i' % batch_idx)

				loss_ = sess.run(
					self.loss,
					feed_dict={self.x: xb, self.y: yb, self.is_training: False})

				val_mse.append((len(xb), loss_))

			val_mse = sum([l * mse for l, mse in val_mse]) / sum([l for l, mse in val_mse])

			val_stats.append((idx, val_mse))

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
		mse = []

		for batch_idx, (xb, yb) in enumerate(self.dataloader.get_batch(
			part, batch_size)):

			y_pred_, mse_ = sess.run(
				[self.y_pred, self.loss],
				feed_dict={self.x: xb, self.y: yb, self.is_training: False})

			y_pred.append(y_pred_)
			y.append(yb)
			mse.append((len(xb), mse_))

		mse = sum([l * mse for l, mse in mse]) / sum([l for l, mse in mse])
		y_pred = np.concatenate(y_pred, axis=0)
		y = np.concatenate(y, axis=0)

		return y_pred, y, mse


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

		features_flat = tf.squeeze(endpoints['global_pool'], [1, 2])

		self.image_features = tf.reshape(
			features_flat,
			(-1, self.n_images, GLOBAL_POOL_N_FEATURES))


	def _build_model(self):

		with tf.compat.v1.variable_scope('image_features'):

			feat = mlp(
				self.image_features,
				[self.image_feature_size],
				dropout_pct=self.dropout_pct,
				activation_fn=self.activation_fn,
				training=self.is_training)

		feature_vec = tf.concat(
			[tf.reduce_mean(feat, axis=1), tf.reduce_max(feat, axis=1)],
			axis=1)

		with tf.compat.v1.variable_scope('outcomes'):

			hidden_layer = mlp(
				feature_vec,
				self.hidden_layer_sizes,
				dropout_pct=self.dropout_pct,
				activation_fn=self.activation_fn,
				training=self.is_training)

			self.y_pred = tf.layers.dense(
				hidden_layer,
				self.n_out,
				activation=None)


	def _build_train_step(self):

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
