import glob
from os import remove, system
from os.path import isfile, join
import shutil

backtranslation_repo_abs_dirname = '/home/tony/QAMetric/Backtranslation/'
backtranslation_script_abs_filename = join(backtranslation_repo_abs_dirname, 'paraphrase_all_languages.py')

def backtranslate_file(DATA_DIR, ANSWER_FILE, OUTPUT_DIR):
	# Check if `ANSWER_FILE` has already been backtranslated
	if isfile(join(OUTPUT_DIR, ANSWER_FILE.split('/')[-1]+'.backtranslations')):
		return 

	# Backtranslate and move backtranslation file into `OUTPUT_DIR`
	system('cd ' + backtranslation_repo_abs_dirname + '; python ' + backtranslation_script_abs_filename + ' ' + ANSWER_FILE)
	shutil.move(ANSWER_FILE+'.backtranslations', OUTPUT_DIR)

	# Delete intermediate backtranslation files
	for f in glob.glob(join(DATA_DIR, '*csv_answers.*')):
		remove(f)

	# Merge backtranslations back into the data file

def backtranslate_cosmosqa():
	DATA_DIR = '/home/tony/answer-generation/data/cosmosqa/'
	DEV_ANSWERS = join(DATA_DIR, 'dev.csv_answers')
	OUTPUT_DIR = 'backtranslation/cosmosqa/'

	backtranslate_file(DATA_DIR, DEV_ANSWERS, OUTPUT_DIR)

def backtranslate_socialiqa():
	DATA_DIR = '/home/tony/answer-generation/data/socialiqa/'
	DEV_ANSWERS = join(DATA_DIR, 'dev.csv_answers')
	TEST_ANSWERS = join(DATA_DIR, 'test.csv_answers')
	OUTPUT_DIR = 'backtranslation/socialiqa/'

	backtranslate_file(DATA_DIR, DEV_ANSWERS, OUTPUT_DIR)
	backtranslate_file(DATA_DIR, TEST_ANSWERS, OUTPUT_DIR)

def backtranslate_narrativeqa():
	DATA_DIR = '/home/tony/answer-generation/data/narrativeqa/'
	DEV_ANSWERS = join(DATA_DIR, 'dev.csv_answers')
	TEST_ANSWERS = join(DATA_DIR, 'test.csv_answers')
	OUTPUT_DIR = 'backtranslation/narrativeqa/'

	backtranslate_file(DATA_DIR, DEV_ANSWERS, OUTPUT_DIR)
	backtranslate_file(DATA_DIR, TEST_ANSWERS, OUTPUT_DIR)

def backtranslate_mcscript():
	DATA_DIR = '/home/tony/answer-generation/data/mcscript/'
	DEV_ANSWERS = join(DATA_DIR, 'dev.csv_answers')
	TEST_ANSWERS = join(DATA_DIR, 'test.csv_answers')
	OUTPUT_DIR = 'backtranslation/mcscript/'

	backtranslate_file(DATA_DIR, DEV_ANSWERS, OUTPUT_DIR)
	backtranslate_file(DATA_DIR, TEST_ANSWERS, OUTPUT_DIR)

if __name__ == '__main__':
	backtranslate_cosmosqa()
	backtranslate_socialiqa()
	backtranslate_narrativeqa()
	backtranslate_mcscript()