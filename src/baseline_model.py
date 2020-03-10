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

	utc = datetime.datetime.utcnow().strftime('%s')

	from data_loader import DataLoader

	dl = DataLoader()

	mdl = BaselineModel(dl)

	with tf.Session() as s:

		train_stats, val_stats = mdl.train(s)

		y_pred, y, mse_all = mdl.predict(sess, 'val')

	mse = np.mean((y - y_pred) ** 2, axis=0)

	import matplotlib.pyplot as plt

	fig, ax = plt.subplots(nrows=1 + dl.n_out, ncols=1, figsize=(5 + 5 * dl.n_out, 5))

	ax[0].plot(*list(zip(*train_stats)), label='train')
	ax[0].plot(*list(zip(*val_stats)), label='val')
	ax[0].set_title('Training Plot')
	ax[0].set_xlabel('Iteration')
	ax[0].set_ylabel('Mean Square Error')
	ax[0].legend()

	for i in range(dl.n_out):

		ax[i + 1].scatter(y[:, i], y_pred[:, i])
		ax[i + 1].set_title(const.OUTCOMES[i] + '(MSE = %.2f)' % mse[i])
		ax[i + 1].set_xlabel('y_true')
		ax[i + 1].set_ylabel('y_pred')

	plt.tight_layout()
	plt.savefig('results_' + utc + '.png')


class BaselineModel:

	def __init__(
		self, dataloader,
		image_feature_size=50,
		hidden_layer_sizes=[],
		learning_rate=1e-3,
		train_mobilenet=False):

		self.dataloader = dataloader

		self.n_out = dataloader.n_out
		self.n_images = dataloader.n_images

		self.image_feature_size = image_feature_size
		self.hidden_layer_sizes = hidden_layer_sizes

		self.learning_rate = learning_rate

		self.train_mobilenet = train_mobilenet

		self._build_placeholders()
		self._build_mobilenet()
		self._build_model()
		self._build_train_step()


	def train(
		self, sess,
		max_epochs=30, max_epochs_no_improve=2,
		batch_size=20, batch_eval_freq=1,
		verbose=True):

		sess.run(tf.global_variables_initializer())
		self.mobilenet_saver.restore(sess, CHECKPOINT_FILE)

		train_stats = []
		val_stats = []
		best_val_nloglik = np.inf
		n_epochs_no_improve = 0

		batches_per_epoch = int(np.ceil(
			self.dataloader.n_train / batch_size))

		xval, yval = self.dataloader.sample_data(part='val', n=-1)

		for epoch_idx in range(max_epochs):

			for batch_idx, (xb, yb) in enumerate(self.dataloader.get_batch(
				'train', batch_size)):

				print('Starting batch %i' % batch_idx)

				loss_, _ = sess.run(
					[self.loss, self.train_step],
					feed_dict={self.x: xb, self.y: yb, self.is_training: True})

				if batch_idx % batch_eval_freq == 0:
					idx = epoch_idx * batches_per_epoch + batch_idx
					train_stats.append((idx, loss_))

			idx = (epoch_idx + 1) * batches_per_epoch
			
			val_stats.append((idx, sess.run(
				self.loss,
				feed_dict={self.x: xval, self.y: yval, self.is_training: False})))

			print('Completed Epoch %i' % epoch_idx)

			if verbose:

				print('Val loss: %.2f, Train loss: %.2f' % (
					val_stats[-1][1],
					np.mean(list(zip(*train_stats[-batches_per_epoch:]))[1])
				))

			if val_stats[-1][1] < best_val_nloglik:
				best_val_nloglik = val_stats[-1][1]
				n_epochs_no_improve = 0
			else:
				n_epochs_no_improve += 1

			if n_epochs_no_improve > max_epochs_no_improve:
				break

		return train_stats, val_stats


	def predict(self, sess, part):

		assert part in ['all', 'train', 'val', 'test']

		x, y = self.dataloader.sample_data(part=part, n=-1)

		y_pred, mse = sess.run(
			[self.y_pred, self.loss],
			feed_dict={self.x: x, self.y: y, self.is_training: False})

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
		self.mobilenet_saver = tf.train.Saver(ema.variables_to_restore())

		features_flat = tf.squeeze(endpoints['global_pool'], [1, 2])

		self.image_features = tf.reshape(
			features_flat,
			(-1, self.n_images, GLOBAL_POOL_N_FEATURES))


	def _build_model(self):

		with tf.variable_scope('image_features'):

			feat = mlp(
				self.image_features,
				[self.image_feature_size],
				training=self.is_training)

		feature_vec = tf.concat(
			[tf.reduce_mean(feat, axis=1), tf.reduce_max(feat, axis=1)],
			axis=1)

		with tf.variable_scope('outcomes'):

			hidden_layer = mlp(
				feature_vec,
				self.hidden_layer_sizes,
				training=self.is_training)

			self.y_pred = tf.layers.dense(
				hidden_layer,
				self.n_out,
				activation=None)


	def _build_train_step(self):

		self.loss = tf.losses.mean_squared_error(
			self.y,
			self.y_pred)

		if self.train_mobilenet:

			self.train_step = tf.train.AdamOptimizer(
				self.learning_rate).minimize(self.loss)

		else:

			myvars = tf.get_collection(
				tf.GraphKeys.GLOBAL_VARIABLES,
				scope='image_features')

			myvars += tf.get_collection(
				tf.GraphKeys.GLOBAL_VARIABLES,
				scope='outcomes')

			self.train_step = tf.train.AdamOptimizer(
				self.learning_rate).minimize(self.loss, var_list=myvars)


def mlp(x, hidden_layer_sizes,
		dropout_pct = 0.,
		activation_fn=tf.nn.relu,
		training=True,
		reuse=False):

	hidden_layer = x

	with tf.variable_scope('mlp', reuse=reuse):

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


if __name__ == '__main__':
	main()
