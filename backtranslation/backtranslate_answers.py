from os import system
from os.path import join

backtranslation_repo_abs_dirname = '/home/tony/QAMetric/Backtranslation/'
backtranslation_script_abs_filename = join(backtranslation_repo_abs_dirname, 'paraphrase_all_languages.py')

def backtranslate_cosmosqa():
	DEV_ANSWERS = '/home/tony/answer-generation/data/cosmosqa/dev.csv_answers'
	TEST_ANSWERS = '/home/tony/answer-generation/data/cosmosqa/test.csv_answers'

	system('cd ' + backtranslation_repo_abs_dirname + '; python ' + backtranslation_script_abs_filename + ' ' + DEV_ANSWERS)
	system('cd ' + backtranslation_repo_abs_dirname + '; python ' + backtranslation_script_abs_filename + ' ' + TEST_ANSWERS)

if __name__ == '__main__':
	backtranslate_cosmosqa()