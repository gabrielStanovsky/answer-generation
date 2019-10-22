import csv
from jsonlines import Reader
from os.path import isfile, join
import random
random.seed(0)

from merge_utils import bert_tokenization_length, match, strip_gpt_endtag, prune_candidates

def load_mcscript_data():
	MCSCRIPT_DEV_FILE = '/home/tony/answer-generation/data/mcscript/dev.csv'
	MCSCRIPT_TEST_FILE = '/home/tony/answer-generation/data/mcscript/test.csv'
	
	data = {}

	with open(MCSCRIPT_DEV_FILE, 'r', encoding='utf8', errors='ignore') as fp:
		fp.readline()   # skip header
		for row in csv.reader(fp):
			context, question, reference = row[1], row[2], row[3]
			if context not in data:
				data[context] = {}
			
			if question not in data[context]:
				data[context][question] = {'reference': reference, 'candidates': set()}

	with open(MCSCRIPT_TEST_FILE, 'r', encoding='utf8', errors='ignore') as fp:
		fp.readline()   # skip header
		for row in csv.reader(fp):
			context, question, reference = row[1], row[2], row[3]
			if context not in data:
				data[context] = {}
			
			if question not in data[context]:
				data[context][question] = {'reference': reference, 'candidates': set()}

	return data

def load_gpt2_predictions(file):
	lines = []

	with open(file, 'r', encoding='utf8', errors='ignore') as fp:
		for row in csv.reader(fp):
			context, question, candidates = row[1], row[2], row[4:]
			candidates = strip_gpt_endtag(candidates)
			lines.append((context, question, candidates))
	return lines

def load_mhpg_predictions(file):
	lines = []
	for line in Reader(open(file)):
		context = line['raw_summary'].replace('\n', ' ').strip()
		question = line['raw_ques'].strip()
		candidate = line['pred'].replace(" n't", "n't").replace(" 's", "'s")

		if 'UNK' not in candidate:
			lines.append((context, question, candidate))
	return lines

def write_data_to_label(data_dict):
	data_list = []

	# First converts the dictionary to entries in a list
	for context in data_dict:
		for question in data_dict[context]:
			reference = data_dict[context][question]['reference']
			candidates = prune_candidates(reference, data_dict[context][question]['candidates'])

			for candidate in candidates:
				# Filter instances that wouldn't fit into BERT
				if bert_tokenization_length(context, question, candidate, reference) < 512 - 4:
					data_list.append([context, question, reference, candidate])

	# Sorts the entries by context
	data_list = sorted(data_list, key = lambda x: x[1])
	data_list = sorted(data_list, key = lambda x: x[0])

	# Write to CSV file
	csvfile = open('merge_predictions/to_label/mcscript.csv', 'w')
	writer = csv.writer(csvfile)
	writer.writerow(['context', 'question', 'reference', 'candidate'])
	for line in data_list:
		writer.writerow(line)
	csvfile.close()

def main():
	# Paths to prediction files
	GPT_PREDICTIONS_DIR = '/home/tony/answer-generation/gpt_models/mcscript'
	GPT2_PREDICTIONS_FILE = [join(GPT_PREDICTIONS_DIR, 'dev.csv_generation'), join(GPT_PREDICTIONS_DIR, 'test.csv_generation')]

	MHPG_PREDICTIONS_DIR = '/home/tony/CommonSenseMultiHopQA/out/mcscript_baseline'
	MHPG_PREDICTIONS_FILE = [join(MHPG_PREDICTIONS_DIR, 'mcscript_valid.jsonl.merged1'), 
							 join(MHPG_PREDICTIONS_DIR, 'mcscript_test.jsonl.merged1')]

	# Load in data and prediction files 	
	data = load_mcscript_data()

	for f in GPT2_PREDICTIONS_FILE:
		for context, question, candidates in load_gpt2_predictions(f):
			data[context][question]['candidates'].update(candidates)

	for f in MHPG_PREDICTIONS_FILE:
		for context, question, candidate in load_mhpg_predictions(f):
			if context in data and question in data[context]:
				data[context][question]['candidates'].add(candidate)

	write_data_to_label(data)

if __name__ == '__main__':
	main()