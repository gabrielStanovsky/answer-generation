import csv
from jsonlines import Reader
from os.path import join
from tqdm import tqdm

from merge_utils import *

def load_socialiqa_data():
	SOCIALIQA_DEV_FILE = '/home/tony/answer-generation/data/socialiqa/dev.csv'
	SOCIALIQA_TEST_FILE = '/home/tony/answer-generation/data/socialiqa/test.csv'
	
	def load_file(SOCIALIQA_FILE, data={}):
		with open(SOCIALIQA_FILE, 'r', encoding='utf8', errors='ignore') as fp:
			fp.readline()   # skip header
			for row in csv.reader(fp):
				context = clean_string(row[1])
				question = clean_string(row[2])
				reference = clean_string(row[3])

				if context not in data:
					data[context] = {}
				
				if question not in data[context]:
					data[context][question] = {'reference': reference, 'candidates': set()}
		return data

	data = load_file(SOCIALIQA_DEV_FILE)
	data = load_file(SOCIALIQA_TEST_FILE, data)

	return data

def load_gpt2_predictions(file):
	lines = []
	with open(file, 'r', encoding='utf8', errors='ignore') as fp:
		for row in csv.reader(fp):
			context = clean_string(row[1])
			question = clean_string(row[2])
			candidates = [clean_string(c) for c in row[4:]]
			lines.append((context, question, candidates))
	return lines

def write_data_to_label(data_dict):
	samples = []

	# First converts the dictionary to entries in a list
	for context in tqdm(data_dict):
		for question in data_dict[context]:
			reference = data_dict[context][question]['reference']
			candidates = prune_candidates(reference, data_dict[context][question]['candidates'])

			for candidate in candidates:
				# Filter instances that wouldn't fit into BERT
				if bert_tokenization_length(context, question, reference, candidate) + 4 > 512:
					continue

				# Check the data instances and get a sample hash id
				hash_id = check_data_and_return_hash(context, question, reference, candidate)
				if hash_id == None:
					continue

				samples.append([context, question, reference, candidate, hash_id])

	samples = prune_and_sort_samples(samples)

	# Write to CSV file
	with open('merge_predictions/to_label/socialiqa.csv', 'w') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['context', 'question', 'reference', 'candidate', 'id'])
		for line in samples:
			writer.writerow(line)

def main():
	# Paths to prediction files
	GPT_PREDICTIONS_DIR = '/home/tony/answer-generation/huggingface_gpt2/models/socialiqa'
	GPT2_PREDICTIONS_FILE = [join(GPT_PREDICTIONS_DIR, 'dev.csv_generation'), join(GPT_PREDICTIONS_DIR, 'test.csv_generation')]


	# Load in data and prediction files 	
	data = load_socialiqa_data()

	for f in GPT2_PREDICTIONS_FILE:
		for context, question, candidates in load_gpt2_predictions(f):
			data[context][question]['candidates'].update(candidates)

	write_data_to_label(data)

if __name__ == '__main__':
	main()