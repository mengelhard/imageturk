import numpy as np
import pandas as pd
import sys, os

'''contains constants and scoring for imageturk data'''


DATA_DIRS = ['/Users/mme/data', '/scratch/mme4']

DATA_SUBDIRS = {
	'smok': ['imageturk_smoker', 'imageturk_smoker_b2'],
	'non': ['imageturk_nonsmoker', 'imageturk_nonsmoker_b2']
}

MODELS_PATHS = [
	'/Users/mme/projects/models/research/slim',
	'/scratch/mme4/models/research/slim'
]

CHECKPOINT_FILE_PATHS = [
	'/Users/mme/projects/imageturk/mobilenet_checkpoint',
	'/scratch/mme4/mobilenet_checkpoint'
]

MOBILENET_OUTPUT_SIZE = {
	'global_pool': 1280,
	'Logits': 1001
}

IMAGES = [
	'QID2', 'QID90', 'QID4', 'QID91', 'QID6',
	'QID92', 'QID7', 'QID93', 'QID10', 'QID96'
]

OUTCOMES = [
	'swan_i',
	'swan_hi',
	'phq',
	'stress',
	'sleep_reg',
	'sleep_dist',
	'eveningness',
	'food_healthiness',
	'food_insecurity',
	'smoking',
	'alcohol',
	'neighborhood_crime',
	'neighborhood_noise',
	'neighborhood_clean',
	'physical_activity',
	'education_level',
	'age',
	'sex',
	'income'
]

# OUTCOMES = ['smoking']
# OUTCOMES = ['age', 'sex', 'income']
OUTCOMES = ['income']

VARTYPES = {
	'swan_i': 'numeric',
	'swan_hi': 'numeric',
	'phq': 'numeric',
	'stress': 'numeric',
	'sleep_reg': 'numeric',
	'sleep_dist': 'numeric',
	'eveningness': 'numeric',
	'food_healthiness': 'numeric',
	'food_insecurity': 'numeric',
	'smoking': 'categorical',
	'alcohol': 'categorical',
	'neighborhood_crime': 'numeric',
	'neighborhood_noise': 'numeric',
	'neighborhood_clean': 'numeric',
	'physical_activity': 'numeric',
	'education_level': 'numeric',
	'age': 'numeric',
	'sex': 'categorical',
	'income': 'numeric'
}

CUTOFFS = {# DIVIDE BY >CUTOFF
	'swan_i': 0,
	'swan_hi': 0,
	'phq': 8,
	'stress': 0,
	'sleep_reg': 2,
	'sleep_dist': 1,
	'eveningness': 1,
	'food_healthiness': 25,
	'food_insecurity': 0,
	'smoking': 0,
	'alcohol': 0,
	'neighborhood_crime': 1,
	'neighborhood_noise': 1,
	'neighborhood_clean': 2,
	'physical_activity': 7,
	'education_level': 3,
	'age': 1985,
	'sex': 1,
	'income': 4
}

ITEMS = {
	'smok': {
		'swan_i': ['QID84_' + str(x + 1) for x in range(9)],
		'swan_hi': ['QID84_' + str(x + 10) for x in range(9)],
		'phq': ['QID17_' + str(x + 1) for x in range(9)],
		'stress_pos': ['QID70_' + str(x) for x in [1, 2, 3, 6, 9, 10]],
		'stress_neg': ['QID70_' + str(x) for x in [4, 5, 7, 8]],
		'sleep_reg': ['QID13'],
		'sleep_dist': ['QID14'],
		'eveningness': ['QID15'],
		'food_healthiness': ['QID21_1', 'QID21_2', 'QID21_3', 'QID23_1',
							 'QID23_2', 'QID23_3', 'QID23_4'],
		'food_insecurity': ['QID25', 'QID26', 'QID27', 'QID28', 'QID29'],
		'smoking': ['smoke'],
		'alcohol': ['QID57_1'],
		'neighborhood_crime': ['QID73_1'],
		'neighborhood_noise': ['QID73_2'],
		'neighborhood_clean': ['QID73_3'],
		'physical_activity': ['QID75_1', 'QID76_1', 'QID77_1'],
		'education_level': ['QID121'],
		'age': ['QID126_TEXT'],
		'sex': ['QID116'],
		'income': ['QID128']
	},
	'non': {
		'swan_i': ['QID84_' + str(x + 1) for x in range(9)],
		'swan_hi': ['QID84_' + str(x + 10) for x in range(9)],
		'phq': ['QID17_' + str(x + 1) for x in range(9)],
		'stress_pos': ['QID70_' + str(x) for x in [1, 2, 3, 6, 9, 10]],
		'stress_neg': ['QID70_' + str(x) for x in [4, 5, 7, 8]],
		'sleep_reg': ['QID13'],
		'sleep_dist': ['QID14'],
		'eveningness': ['QID15'],
		'food_healthiness': ['QID21_1', 'QID21_2', 'QID21_3', 'QID23_1',
							 'QID23_2', 'QID23_3', 'QID23_4'],
		'food_insecurity': ['QID25', 'QID26', 'QID27', 'QID28', 'QID29'],
		'smoking': ['smoke'],
		'alcohol': ['QID57_1'],
		'neighborhood_crime': ['QID73_1'],
		'neighborhood_noise': ['QID73_2'],
		'neighborhood_clean': ['QID73_3'],
		'physical_activity': ['QID75_1', 'QID76_1', 'QID77_1'],
		'education_level': ['QID121'],
		'age': ['QID115_TEXT'],
		'sex': ['QID116'],
		'income': ['QID123']
	}
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

PHQ_SCALE = {
	'Not at all': 0,
	'Several days': 1,
	'More than half the days': 2,
	'Nearly every day': 3
}

STRESS_SCALE = {
	'Never': 0,
	'Almost Never': 1,
	'Sometimes': 2,
	'Fairly Often': 3,
	'Very Often': 4
}

SLEEP_LIKERT_SCALE = {
	'Never': 0,
	'Almost Never': 1,
	'Sometimes': 2,
	'Almost Always': 3,
	'Always': 4
}

EVENINGNESS_SCALE = {
	'Definitely a morning type': 0,
	'More a morning type than evening type': 1,
	'More an evening type than morning type': 2,
	'Definitely an evening type': 3
}

HEALTHINESS_SCALE = {
	'Strongly Disagree': 0,
	'Disagree': 1,
	'Somewhat Disagree': 2,
	'Neither Agree nor Disagree': 3,
	'Somewhat Agree': 4,
	'Agree': 5,
	'Strongly Agree': 6
}

INSECURITY_SCALE = {
	'Never True': 0,
	'Sometimes True': 1,
	'Often True': 2,
	'No': 0,
	'Yes': 2,
	'Yes, only 1 or 2 months': 1,
	'Yes, some months but not every month': 2,
	'Yes, almost every month': 3
}

SMOKING_SCALE = {
	'No': 0,
	'Yes': 1
}

EDUCATION_SCALE = {
	'8th grade or less': 0,
	'Some high school': 1,
	'High school diploma': 2,
	'High school GED': 2,
	'Some college but no degree': 3,
	'Associate degree - occupations/vocational': 3,
	'Associate degree - academic program':3,
	'Bachelor\'s degree (e.g. BA, AB, BS)': 4,
	'Master\'s degree (e.g. MA, MS, MENG, M.ED., MSW)': 5,
	'Professional degree (e.g. MD, DDS, DVM, JD)': 5,
	'Doctorate degree (e.g. PhD, EdD)': 5
}

NEIGHBORHOOD_SCALE = {
	'Strongly Disagree': 0,
	'Disagree': 1,
	'Neutral (Neither Agree nor Disagree)': 2,
	'Agree': 3,
	'Strongly Agree': 4
}

SEX_SCALE = {
	'Female': 0,
	'Prefer not to say': 1,
	'Male': 2
}

INCOME_SCALE = {
	'Less than $12,000': 0,
	'$12,000 to $15,999': 1,
	'$16,000 to $24,999': 2,
	'$25,000 to $34,999': 3,
	'$35,000 to $49,999': 4,
	'$50,000 to $74,999': 5,
	'$75,000 to $99,999': 6,
	'$100,000 to $149,999': 7,
	'$150,000 to $199,999': 8,
	'$200,000 or more': 9
}

SCALES = {
	'swan_i': SWAN_SCALE,
	'swan_hi': SWAN_SCALE,
	'phq': PHQ_SCALE,
	'stress_pos': STRESS_SCALE,
	'stress_neg': STRESS_SCALE,
	'sleep_reg': SLEEP_LIKERT_SCALE,
	'sleep_dist': SLEEP_LIKERT_SCALE,
	'eveningness': EVENINGNESS_SCALE,
	'food_healthiness': HEALTHINESS_SCALE,
	'food_insecurity': INSECURITY_SCALE,
	'smoking': SMOKING_SCALE,
	'neighborhood_crime': NEIGHBORHOOD_SCALE,
	'neighborhood_noise': NEIGHBORHOOD_SCALE,
	'neighborhood_clean': NEIGHBORHOOD_SCALE,
	'education_level': EDUCATION_SCALE,
	'sex': SEX_SCALE,
	'income': INCOME_SCALE
}


def item_total_categorical(item_frame, item_scale):

	return item_frame.applymap(
		lambda x: item_scale.get(x, x)).sum(axis=1).values


def item_total_numeric(item_frame):

	return item_frame.sum(axis=1).values


def score_outcome(df, group, outcome, dichotomize=None):

	if outcome == 'stress':

		s = score_outcome(
			df, group, 'stress_pos', dichotomize=False) - score_outcome(
			df, group, 'stress_neg', dichotomize=False)

	elif outcome in ['alcohol', 'physical_activity', 'age']:

		s = item_total_numeric(df[ITEMS[group][outcome]])

	else:

		s = item_total_categorical(df[ITEMS[group][outcome]], SCALES[outcome])

	if dichotomize == True:

		return (s > CUTOFFS[outcome]).astype(float)

	elif dichotomize == False:

		return s

	elif VARTYPES[outcome] == 'categorical':

		return (s > CUTOFFS[outcome]).astype(float)

	elif VARTYPES[outcome] == 'numeric':

		return s

	else:

		assert False, 'Could not determine variable type for %s' % outcome
