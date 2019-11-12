import glob
from os import remove, system
from os.path import isfile, join
import shutil

backtranslation_repo_abs_dirname = '/home/tony/QAMetric/Backtranslation/'
backtranslation_script_abs_filename = join(backtranslation_repo_abs_dirname, 'paraphrase_all_languages.py')

def backtranslate_cosmosqa():
	DATA_DIR = '/home/tony/answer-generation/data/cosmosqa/'
	TRAIN_ANSWERS = join(DATA_DIR, 'train.csv_answers')
	DEV_ANSWERS = join(DATA_DIR, 'dev.csv_answers')

	if isfile(join(DATA_DIR, 'train.csv_answers.backtranslations')) and \
		isfile(join(DATA_DIR, 'dev.csv_answers.backtranslations')):
		return 

	system('cd ' + backtranslation_repo_abs_dirname + '; python ' + backtranslation_script_abs_filename + ' ' + TRAIN_ANSWERS)
	system('cd ' + backtranslation_repo_abs_dirname + '; python ' + backtranslation_script_abs_filename + ' ' + DEV_ANSWERS)

	# Move backtranslation files into the backtranslation directory
	shutil.move(join(DATA_DIR, 'train.csv_answers.backtranslations'), 'backtranslation/cosmosqa/')
	shutil.move(join(DATA_DIR, 'dev.csv_answers.backtranslations'), 'backtranslation/cosmosqa/')

	# Delete intermediate backtranslation files
	for f in glob.glob(join(DATA_DIR, '*csv_answers.*')):
		remove(f)

def backtranslate_socialiqa():
	DATA_DIR = '/home/tony/answer-generation/data/socialiqa/'
	TRAIN_ANSWERS = join(DATA_DIR, 'train.csv_answers')
	DEV_ANSWERS = join(DATA_DIR, 'dev.csv_answers')
	TEST_ANSWERS = join(DATA_DIR, 'test.csv_answers')

	if isfile(join(DATA_DIR, 'train.csv_answers.backtranslations')) and \
		isfile(join(DATA_DIR, 'dev.csv_answers.backtranslations')) and \
		isfile(join(DATA_DIR, 'test.csv_answers.backtranslations')):
		return 
		
	system('cd ' + backtranslation_repo_abs_dirname + '; python ' + backtranslation_script_abs_filename + ' ' + TRAIN_ANSWERS)
	system('cd ' + backtranslation_repo_abs_dirname + '; python ' + backtranslation_script_abs_filename + ' ' + DEV_ANSWERS)
	system('cd ' + backtranslation_repo_abs_dirname + '; python ' + backtranslation_script_abs_filename + ' ' + TEST_ANSWERS)

	# Move backtranslation files into the backtranslation directory
	shutil.move(join(DATA_DIR, 'train.csv_answers.backtranslations'), 'backtranslation/socialiqa/')
	shutil.move(join(DATA_DIR, 'dev.csv_answers.backtranslations'), 'backtranslation/socialiqa/')
	shutil.move(join(DATA_DIR, 'test.csv_answers.backtranslations'), 'backtranslation/socialiqa/')

	# Delete intermediate backtranslation files
	for f in glob.glob(join(DATA_DIR, '*csv_answers.*')):
		remove(f)

def backtranslate_narrativeqa():
	pass

def backtranslate_mcscript():
	pass

if __name__ == '__main__':
	backtranslate_cosmosqa()
	# backtranslate_socialiqa()
	# backtranslate_narrativeqa()
	# backtranslate_mcscript()