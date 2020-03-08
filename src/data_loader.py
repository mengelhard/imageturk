import numpy as np
import pandas as pd
import ast
import os

import constants as const


def main():

	dl = DataLoader()

	print('Total participants:', len(dl.data['all']))

	print('Train data:')
	print(dl.data['train'][const.OUTCOMES])

	print('Val data:')
	print(dl.data['val'][const.OUTCOMES])

	print('Test data:')
	print(dl.data['test'][const.OUTCOMES])

	print('Testing batches:')
	for batch_x, batch_y in dl.get_batch('train', 50):
		print(batch_x)
		print(batch_y)


class DataLoader:

	def __init__(self, data_partition=[.6, .8]):

		self.datadir = os.path.join(
			check_directories(const.DATA_DIRS),
			'imageturk')

		df_smok = self._get_datafile('smok')
		df_non = self._get_datafile('non')

		data_smok = self._get_image_filenames(df_smok, 'smok').join(
			self._get_outcomes(df_smok, 'smok'))

		data_non = self._get_image_filenames(df_non, 'non').join(
			self._get_outcomes(df_non, 'non'))

		self.data = dict()

		self.data['all'] = pd.concat([data_smok, data_non], axis=0).sample(
			frac=1, random_state=0) # shuffle rows

		vidx, tidx = (int(x * len(self.data['all'])) for x in data_partition)

		self.data['train'] = self.data['all'].iloc[:vidx, :]
		self.data['val'] = self.data['all'].iloc[vidx:tidx, :]
		self.data['test'] = self.data['all'].iloc[tidx:, :]


	def get_batch(self, part, batch_size):

		assert part in ['all', 'train', 'val', 'test']
		
		l = len(self.data[part])

		for ndx in range(0, l, batch_size):

			endx = min(ndx + batch_size, l)

			data = self.data[part].iloc[ndx:endx, :]

			filenames = data.drop(const.OUTCOMES, axis=1).values
			outcomes = data[const.OUTCOMES].values

			yield filenames, outcomes


	def _get_datafile(self, group):

		subdir = const.DATA_SUBDIR[group]

		fns = listdir_by_ext(
			os.path.join(self.datadir, subdir),
			'.csv')

		print('Found %i .csv files in' % len(fns), subdir)
		print('Reading', fns[-1])

		return self._filter_imageturk_csv(
			*self._read_imageturk_csv(
				os.path.join(self.datadir, subdir, fns[-1])),
			group
			)


	def _read_imageturk_csv(self, fn):

		df = pd.read_csv(fn, header=[0, 1, 2])
		
		colnames = [ast.literal_eval(x[2])['ImportId'] for x in df.columns.values]

		assert len(colnames) == len(set(colnames))

		coltext = [x[1] for x in df.columns.values]
		coldict = {y:x for x, y in zip(colnames, coltext)}

		df.columns = colnames

		return df.set_index('_recordId'), coldict


	def _filter_imageturk_csv(self, df, coldict, group):

		imagecols = get_filecols(df)
		sizecols = [x.split('_')[0] + '_FILE_SIZE' for x in imagecols]
		imagesizes = np.array([df[x].fillna(1e6).values for x in sizecols])

		exclusion_criteria = [
			df['distributionChannel'] == 'preview',
			~df['finished'],
			df[coldict['At birth, were you described as:']].isna(),
			np.any(imagesizes < 1e5, axis=0)
		]

		fdf = df[~np.any(exclusion_criteria, axis=0)]

		#print('The following columns have null values:')
		#print(fdf.columns[fdf.isnull().any()].values)

		#print(fdf.isna().sum().reset_index().values)

		return fdf


	def _get_outcomes(self, df, group):
		return pd.DataFrame(
			{o: const.score_scales[o](df, group) for o in const.OUTCOMES},
			index=df.index)


	def _get_image_filenames(self, df, group):

		cols = const.IMAGES[group]

		if group == 'smok':
			basedir = os.path.join(self.datadir, 'imageturk_smoker')
		elif group == 'non':
			basedir = os.path.join(self.datadir, 'imageturk_nonsmoker')

		return pd.DataFrame(
			{c: basedir + '/' + c + '/' + df[c + '_FILE_ID'] + '~' + df[c + '_FILE_NAME']
			 for c in cols},
			index=df.index)


def check_directories(dirlist):

	for i, d in enumerate(dirlist):

		if os.path.exists(d):

			print('Found data directory', d)

			break

		if (i + 1) == len(dirlist):

			print('No data directory found')

			assert False

	return d


def listdir_by_ext(directory, extension=None):

	if extension is None:

		return os.listdir(directory)

	else:

		return [x for x in os.listdir(directory)
				if os.path.splitext(x)[-1] == extension]


def get_filecols(df):

	return [x for x in df.columns.values if (x[-7:] == 'FILE_ID')]


if __name__ == '__main__':
	main()

