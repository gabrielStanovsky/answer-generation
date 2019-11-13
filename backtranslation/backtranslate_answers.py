import csv
import glob
from os import remove, system
from os.path import isfile, join
import shutil

backtranslation_repo_abs_dirname = '/home/tony/QAMetric/Backtranslation/'
backtranslation_script_abs_filename = join(backtranslation_repo_abs_dirname, 'paraphrase_all_languages.py')

def backtranslate_file(DATA_DIR, ANSWER_FILE, OUTPUT_DIR):
	output_file = join(OUTPUT_DIR, ANSWER_FILE.split('/')[-1]+'.backtranslations')

	# Check if `ANSWER_FILE` has already been backtranslated
	if isfile(output_file):
		return

	# Backtranslate
	system('cd ' + backtranslation_repo_abs_dirname + '; python ' + backtranslation_script_abs_filename + ' ' + ANSWER_FILE)

	# Merge backtranslations back into the data file
	bt_file = open(ANSWER_FILE+'.backtranslations')
	data_file = open(join('data', ANSWER_FILE[:-8]))
	csvfile = open(output_file, 'w')
	writer = csv.writer(csvfile)

	# Skip header line
	data_file.readline()

	# Iterate through data and bt file, append bt to data_line
	for data_line, bt_line in zip(csv.reader(data_file), bt_file):
		bt_reference = bt_line.split('\t')[0].replace('"','').replace("'", "")
		data_reference = [e.replace('"','').replace("'", "") for e in data_line[3:]]

		print(bt_reference, data_reference, '\n')
		# Check that data and bt files are aligned correctly
		assert bt_reference in data_reference
		
		bts = eval(bt_line.split('\t')[-1])
		data_line += bts
		writer.writerow(data_line)

	bt_file.close()
	data_file.close()
	csvfile.close()

	# Delete intermediate backtranslation files
	for f in glob.glob(join(DATA_DIR, '*csv_answers.*')):
		remove(f)

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