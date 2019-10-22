from collections import Counter 
import csv
import json
from os.path import isfile, join
import random
random.seed(0)

from merge_utils import bert_tokenization_length, match, prune_candidates

def most_frequent(l): 
	occurence_count = Counter(l) 
	return occurence_count.most_common(1)[0][0] 

def load_ropes_data():
	ROPES_DEV_FILE = '/home/tony/answer-generation/raw_data/ropes/dev-v0.4.json'
	ROPES_TEST_FILE = '/home/tony/answer-generation/raw_data/ropes/test-v0.4.json'

	data = {}
	
	for file in [ROPES_DEV_FILE, ROPES_TEST_FILE]:
		with open(file) as dataset_file:
			dataset = json.load(dataset_file)

		for line in dataset['data'][0]['paragraphs']:
			context = line['context'].split('[SEP]')[0].strip().replace('\n', ' ').replace('__SEP__', ' ')
			question = line['context'].split('[SEP]')[1].strip().replace('\n', ' ')

			for question_dict in line['qas']:
				question_id = str(question_dict['id'])
				reference = question_dict['answers'][0]['text']

				data[question_id] = {'context': context, 'question': question, 'reference': reference, 'candidates': set()}

	return data

def load_predictions(file):
	with open(file) as dataset_file:
		predictions = json.load(dataset_file)

	return predictions

def write_data_to_label(data_dict):
	data_list = []

	# First converts the dictionary to entries in a list
	for question_id in data_dict:
		context = data_dict[question_id]['context']
		question = data_dict[question_id]['question']
		reference = data_dict[question_id]['reference']

		candidates = prune_candidates(reference, data_dict[question_id]['candidates'])

		for candidate in candidates:
			# Filter instances that wouldn't fit into BERT
			if bert_tokenization_length(context, question, candidate, reference) < 512 - 4:
				data_list.append([context, question, reference, candidate])

	# Sorts the entries by context
	data_list = sorted(data_list, key = lambda x: x[3])
	data_list = sorted(data_list, key = lambda x: x[1])
	data_list = sorted(data_list, key = lambda x: x[0])

	# Write to CSV file
	csvfile = open('merge_predictions/to_label/ropes.csv', 'w')
	writer = csv.writer(csvfile)
	writer.writerow(['context', 'question', 'reference', 'candidate'])
	for line in data_list:
		writer.writerow(line)
	csvfile.close()

def main():
	# Paths to prediction files
	PREDICTION_FILES = ['/home/tony/answer-generation/bert_models/ropes/nbest_dev_predictions.json', \
						'/home/tony/answer-generation/bert_models/ropes/nbest_test_predictions.json']

	# Load in data and prediction files 	
	data = load_ropes_data()

	for f in PREDICTION_FILES:
		for question_id, candidate in load_predictions(f).items():
			data[question_id]['candidates'].update(candidate)

	write_data_to_label(data)

if __name__ == '__main__':
	main()