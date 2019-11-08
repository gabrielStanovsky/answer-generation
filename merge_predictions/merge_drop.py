from allennlp.data.dataset_readers.reading_comprehension.drop import DropReader
from collections import Counter 
import csv
import json
from os.path import isfile, join
import random
random.seed(0)

from merge_utils import *

def most_frequent(l): 
	occurence_count = Counter(l) 
	return occurence_count.most_common(1)[0][0] 

def load_drop_data():
	DROP_DEV_FILE = '/home/tony/answer-generation/raw_data/drop/drop_dataset_dev.json'
	
	data = {}

	with open(DROP_DEV_FILE) as dataset_file:
		dataset = json.load(dataset_file)
	
	for passage_id, passage_info in dataset.items():
		passage = clean_string(passage_info["passage"])
		for question_answer in passage_info["qa_pairs"]:
			question_id = question_answer["query_id"]
			question = clean_string(question_answer["question"])
			answer_annotations = []
			if "answer" in question_answer:
				answer_annotations.append(question_answer["answer"])
			if "validated_answers" in question_answer:
				answer_annotations += question_answer["validated_answers"]
			
			# Extract out the label per annotation
			answer_annotations = [' '.join(DropReader.extract_answer_info_from_annotation(a)[1]) for a in answer_annotations]
			
			# Get the most common answer as the gold answer
			reference = clean_string(str(most_frequent(answer_annotations)))
			
			data[question_id] = {'context': passage, 'question': question, 'reference': reference, 'candidates': set()}

	return data

def load_predictions(file):
	with open(file) as dataset_file:
		return json.load(dataset_file)

def write_data_to_label(data_dict):
	samples = []

	# First converts the dictionary to entries in a list
	for question_id in data_dict:
		context = data_dict[question_id]['context']
		question = data_dict[question_id]['question']
		reference = data_dict[question_id]['reference']
		candidates = prune_candidates(reference, data_dict[question_id]['candidates'])

		for candidate in candidates:
			# Filter instances that wouldn't fit into BERT
			if bert_tokenization_length(context, question, candidate, reference) + 4 > 512:
				continue

			# Check the data instances and get a sample hash id
			hash_id = check_data_and_return_hash(context, question, reference, candidate)
			if hash_id == None:
				continue

			samples.append([context, question, reference, candidate, hash_id])

	samples = prune_and_sort_samples(samples)

	# Write to CSV file
	with open('merge_predictions/to_label/drop.csv', 'w') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['context', 'question', 'reference', 'candidate'])
		for line in samples:
			writer.writerow(line)

def main():
	# Paths to prediction files
	PREDICTION_FILES = ['/home/tony/answer-generation/nabert/models/drop/best_dev_pred.json', \
						'/home/tony/answer-generation/naqanet/models/drop/best_dev_pred.json']

	# Load in data and prediction files 	
	data = load_drop_data()

	for f in PREDICTION_FILES:
		for question_id, candidate in load_predictions(f).items():
			data[question_id]['candidates'].add(str(candidate))

	write_data_to_label(data)

if __name__ == '__main__':
	main()