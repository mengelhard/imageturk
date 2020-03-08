import numpy as np
import pandas as pd

'''contains constants and scoring for imageturk data'''


DATA_DIRS = ['/Users/mme/data']

DATA_SUBDIR = {
	'smok': 'imageturk_smoker',
	'non': 'imageturk_nonsmoker'
}

OUTCOMES = ['swan_i', 'swan_hi', 'phq', 'stress', 'smok']

SWAN_I = {
	'non': ['QID84_' + str(x + 1) for x in range(9)],
	'smok': ['QID84_' + str(x + 1) for x in range(9)]
}

SWAN_HI = {
	'non': ['QID84_' + str(x + 10) for x in range(9)],
	'smok': ['QID84_' + str(x + 10) for x in range(9)]
}

SWAN_SCALE = {
	'Far Above': -3,
	'Above': -2,
	'Slightly Above': -1,
	'Average': 0,
	'Slightly Below': 1,
	'Below': 2,
	'Far Below': 3
}

PHQ = {
	'non': ['QID17_' + str(x + 1) for x in range(9)],
	'smok': ['QID17_' + str(x + 1) for x in range(9)]
}

PHQ_SCALE = {
	'Not at all': 0,
	'Several days': 1,
	'More than half the days': 2,
	'Nearly every day': 3
}

STRESS = {
	'non': {
		'pos': ['QID70_' + str(x) for x in [1, 2, 3, 6, 9, 10]],
		'neg': ['QID70_' + str(x) for x in [4, 5, 7, 8]]},
	'smok': {
		'pos': ['QID70_' + str(x) for x in [1, 2, 3, 6, 9, 10]],
		'neg': ['QID70_' + str(x) for x in [4, 5, 7, 8]]}
}

STRESS_SCALE = {
	'Never': 0,
	'Almost Never': 1,
	'Sometimes': 2,
	'Fairly Often': 3,
	'Very Often': 4
}

IMAGES = {
	'non': [
		'QID2', 'QID90', 'QID4', 'QID91', 'QID6',
		'QID92', 'QID7', 'QID93', 'QID10', 'QID96'],
	'smok': [
		'QID2', 'QID90', 'QID4', 'QID91', 'QID6',
		'QID92', 'QID7', 'QID93', 'QID10', 'QID96']
}


def score_swan_i(df, group):
	return df[SWAN_I[group]].applymap(
		lambda x: SWAN_SCALE[x]).sum(axis=1).values


def score_swan_hi(df, group):
	return df[SWAN_HI[group]].applymap(
		lambda x: SWAN_SCALE[x]).sum(axis=1).values


def score_phq(df, group):
	return df[PHQ[group]].applymap(
		lambda x: PHQ_SCALE[x]).sum(axis=1).values


def score_stress(df, group):
	pos = df[STRESS[group]['pos']].applymap(
		lambda x: STRESS_SCALE[x]).sum(axis=1).values
	neg = df[STRESS[group]['neg']].applymap(
		lambda x: 4 - STRESS_SCALE[x]).sum(axis=1).values
	return pos + neg


def score_smok(df, group):
	if group == 'non':
		return np.zeros(len(df))
	elif group == 'smok':
		return np.ones(len(df))


score_scales = {
	'swan_i': score_swan_i,
	'swan_hi': score_swan_hi,
	'phq': score_phq,
	'stress': score_stress,
	'smok': score_smok
}
