import csv
import glob
from os import remove, system
from os.path import isfile, join
import shutil

masked_lm_repo_abs_dirname = '/home/tony/QAMetric/Masked-Answer-Paraphrase-Generation'
masked_lm_script_abs_filename = join(masked_lm_repo_abs_dirname, 'generate_answer_paraphrases.py')

def create_masked_lm_paraphrases(DATA_DIR, DATA_FILE, OUTPUT_DIR):
	output_file = join(OUTPUT_DIR, DATA_FILE + '.masked_lm')

	# Check if `DATA_FILE` has already has masked lm paraphrases generated
	if isfile(output_file):
		return

	# Generate masked lm answer candidates
	cmd = 'cd ' + masked_lm_repo_abs_dirname + '; python ' + masked_lm_script_abs_filename + \
			' -p ' + DATA_DIR + ' -d ' + DATA_FILE + '_question_answers' + ' -c ' + DATA_FILE + '_context'
	print(output_file)
	print(cmd)
	system(cmd)


def masked_lm_cosmosqa():
	DATA_DIR = '/home/tony/answer-generation/data/cosmosqa/'
	TRAIN_FILE = 'train.csv'
	DEV_FILE = 'dev.csv'
	OUTPUT_DIR = 'masked_lm/cosmosqa/'

	# create_masked_lm_paraphrases(DATA_DIR, TRAIN_FILE, OUTPUT_DIR)
	create_masked_lm_paraphrases(DATA_DIR, DEV_FILE, OUTPUT_DIR)

def masked_lm_socialiqa():
	DATA_DIR = '/home/tony/answer-generation/data/socialiqa/'
	TRAIN_FILE = 'train.csv'
	DEV_FILE = 'dev.csv'
	TEST_FILE = 'test.csv'
	OUTPUT_DIR = 'masked_lm/socialiqa/'

	# create_masked_lm_paraphrases(DATA_DIR, TRAIN_FILE, OUTPUT_DIR)
	create_masked_lm_paraphrases(DATA_DIR, DEV_FILE, OUTPUT_DIR)
	create_masked_lm_paraphrases(DATA_DIR, TEST_FILE, OUTPUT_DIR)

def masked_lm_narrativeqa():
	DATA_DIR = '/home/tony/answer-generation/data/narrativeqa/'
	TRAIN_FILE = 'train.csv'
	DEV_FILE = 'dev.csv'
	TEST_FILE = 'test.csv'
	OUTPUT_DIR = 'masked_lm/narrativeqa/'

	# create_masked_lm_paraphrases(DATA_DIR, TRAIN_FILE, OUTPUT_DIR)
	create_masked_lm_paraphrases(DATA_DIR, DEV_FILE, OUTPUT_DIR)
	create_masked_lm_paraphrases(DATA_DIR, TEST_FILE, OUTPUT_DIR)

def masked_lm_mcscript():
	DATA_DIR = '/home/tony/answer-generation/data/mcscript/'
	TRAIN_FILE = 'train.csv'
	DEV_FILE = 'dev.csv'
	TEST_FILE = 'test.csv'
	OUTPUT_DIR = 'masked_lm/mcscript/'

	# create_masked_lm_paraphrases(DATA_DIR, TRAIN_FILE, OUTPUT_DIR)
	create_masked_lm_paraphrases(DATA_DIR, DEV_FILE, OUTPUT_DIR)
	create_masked_lm_paraphrases(DATA_DIR, TEST_FILE, OUTPUT_DIR)

if __name__ == '__main__':
	masked_lm_cosmosqa()
	# masked_lm_socialiqa()
	# masked_lm_narrativeqa()
	# masked_lm_mcscript()