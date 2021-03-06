import csv
from jsonlines import Reader
from os.path import join
from tqdm import tqdm

from merge_utils import *

def load_cosmosqa_data():
	COSMOSQA_DEV_FILE = '/home/tony/answer-generation/data/cosmosqa/dev.csv'
	
	def load_file(COSMOSQA_FILE, data={}):
		with open(COSMOSQA_FILE, 'r', encoding='utf8', errors='ignore') as fp:
			fp.readline()   # skip header
			for row in csv.reader(fp):
				context = clean_string(row[1])
				question = clean_string(row[2])
				reference = clean_string(row[3])

				if context not in data:
					data[context] = {}
				
				if question not in data[context]:
					data[context][question] = {'reference': reference, 'candidates': {}}
		return data

	data = load_file(COSMOSQA_DEV_FILE)

	return data

def load_gpt2_predictions(file):
	lines = []
	with open(file, 'r', encoding='utf8', errors='ignore') as fp:
		for row in csv.reader(fp):
			context = clean_string(row[1])
			question = clean_string(row[2])
			candidates = {clean_string(c) : 'gpt2' for c in row[4:]}
			lines.append((context, question, candidates))
	return lines

def load_backtranslations(file):
	lines = []
	with open(file, 'r', encoding='utf8', errors='ignore') as fp:
		for row in csv.reader(fp):
			context = clean_string(row[1])
			question = clean_string(row[2])

			# c[0] because c[1] is the score of the backtranslation
			candidates = {clean_string(c) : 'backtranslation' for c in row[4:]}
			lines.append((context, question, candidates))
	return lines

def write_data_to_label(data_dict):
	samples = []

	# First converts the dictionary to entries in a list
	for context in tqdm(data_dict):
		for question in data_dict[context]:
			reference = data_dict[context][question]['reference']
			candidates = data_dict[context][question]['candidates']
			unique_candidates_keys = prune_candidates(reference, candidates)

			for candidate_key in unique_candidates_keys:
				# Filter instances that wouldn't fit into BERT
				if bert_tokenization_length(context, question, reference, candidate_key) + 4 > 512:
					continue

				# Check the data instances and get a sample hash id
				hash_id = check_data_and_return_hash(context, question, reference, candidate_key)
				if hash_id == None:
					continue

				samples.append([context, question, reference, candidate_key, candidates[candidate_key], hash_id])

	samples = prune_and_sort_samples(samples)

	# Write to CSV file
	with open('merge_predictions/merged_datasets/cosmosqa.csv', 'w') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['context', 'question', 'reference', 'candidate', 'source', 'id'])
		for line in samples:
			writer.writerow(line)

def main():
	# Paths to prediction files
	GPT_PREDICTIONS_DIR = '/home/tony/answer-generation/huggingface_gpt2/models/cosmosqa'
	GPT2_PREDICTIONS_FILE = join(GPT_PREDICTIONS_DIR, 'dev.csv_generation')

	BACKTRANSLATION_DIR = '/home/tony/answer-generation/backtranslation/cosmosqa'
	BACKTRANSLATION_FILE = join(BACKTRANSLATION_DIR, 'dev.csv_answers.backtranslations.filtered')

	# Load in data and prediction files 	
	data = load_cosmosqa_data()

	for context, question, candidates in load_gpt2_predictions(GPT2_PREDICTIONS_FILE):
		data[context][question]['candidates'].update(candidates)

	for context, question, candidate in load_backtranslations(BACKTRANSLATION_FILE):
		data[context][question]['candidates'].update(candidate)

	write_data_to_label(data)

if __name__ == '__main__':
	main()