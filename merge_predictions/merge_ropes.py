import csv
import json
from tqdm import tqdm

from merge_utils import *

def load_ropes_data():
	ROPES_DEV_FILE = '/home/tony/answer-generation/raw_data/ropes/dev-v0.4.json'
	ROPES_TEST_FILE = '/home/tony/answer-generation/raw_data/ropes/test-v0.4.json'

	data = {}
	
	for file in [ROPES_DEV_FILE, ROPES_TEST_FILE]:
		with open(file) as dataset_file:
			dataset = json.load(dataset_file)

		for line in dataset['data'][0]['paragraphs']:
			context = clean_string(line['context'].split('[SEP]')[0]).replace('__SEP__', ' ')
			question = clean_string(line['context'].split('[SEP]')[1])

			for question_dict in line['qas']:
				question_id = str(question_dict['id'])
				reference = clean_string(question_dict['answers'][0]['text'])

				data[question_id] = {'context': context, 'question': question, 'reference': reference, 'candidates': {}}

	return data

def load_predictions(file):
	with open(file) as dataset_file:
		return json.load(dataset_file)

def write_data_to_label(data_dict):
	samples = []

	# First converts the dictionary to entries in a list
	for question_id in tqdm(data_dict):
		context = data_dict[question_id]['context']
		question = data_dict[question_id]['question']
		reference = data_dict[question_id]['reference']
		candidates = data_dict[question_id]['candidates']
		unique_candidates_keys = prune_candidates(reference, candidates)

		for candidate_key in unique_candidates_keys:
			if 'SEP' in candidate_key:
				continue

			# Filter instances that wouldn't fit into BERT
			if bert_tokenization_length(context, question, candidate_key, reference) + 4 > 512:
				continue

			# Check the data instances and get a sample hash id
			hash_id = check_data_and_return_hash(context, question, reference, candidate_key)
			if hash_id == None:
				continue

			samples.append([context, question, reference, candidate_key, candidates[candidate_key], hash_id])


	samples = prune_and_sort_samples(samples)

	# Write to CSV file
	with open('merge_predictions/merged_datasets/ropes.csv', 'w') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['context', 'question', 'reference', 'candidate', 'source', 'id'])
		for line in samples:
			writer.writerow(line)

def main():
	# Paths to prediction files
	PREDICTION_FILES = ['/home/tony/answer-generation/bert/models/ropes/nbest_dev_predictions.json', \
						'/home/tony/answer-generation/bert/models/ropes/nbest_test_predictions.json']

	# Load in data and prediction files 	
	data = load_ropes_data()

	for f in PREDICTION_FILES:
		for question_id, candidates in load_predictions(f).items():
			data[question_id]['candidates'].update({c: 'bert' for c in candidates})

	write_data_to_label(data)

if __name__ == '__main__':
	main()