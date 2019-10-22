import csv
from jsonlines import Reader
from os.path import join
import random
import spacy
from spacy.lang.en.stop_words import STOP_WORDS 
import string

nlp = spacy.load('en_core_web_sm', disable=['parser','ner', 'tagger'])	
STOP_WORDS.update(string.punctuation)
random.seed(0)

from merge_utils import *

def are_two_answers_the_same(question, ref1, ref2):
	""" 
	Used to determine if two answers (from the same question) are the same via n-gram overlap.
	This is useful for cases like MultiRC where there are multiple correct answers but 
	it isn't clear whether they are the same. 
	"""
	question = [token.lemma_ for token in nlp(question.lower())]
	ref1 	 = [token.lemma_ for token in nlp(ref1.lower())]
	ref2 	 = [token.lemma_ for token in nlp(ref2.lower())]

	stripped_question = [token for token in question if token not in STOP_WORDS]
	stripped_ref1 	  = [token for token in ref1 if token not in STOP_WORDS or token in stripped_question]
	stripped_ref2 	  = [token for token in ref2 if token not in STOP_WORDS or token in stripped_question]

	# Get overlap between ans1 and ans2
	token_overlap = len(set(stripped_ref1) & set(stripped_ref2))
	shorter_ans_len =  min(len(stripped_ref1), len(stripped_ref2))

	if shorter_ans_len == 0 or token_overlap/shorter_ans_len >= 0.5:
		if len(ref1) >= len(ref2) or 'and' not in ref2 or 'and' in ref1:
			return True

	return False

def load_narrativeqa_data():
	NARRATIVEQA_DEV_FILE = '/home/tony/answer-generation/data/narrativeqa/dev.csv'
	NARRATIVEQA_TEST_FILE = '/home/tony/answer-generation/data/narrativeqa/test.csv'

	def load_file(NARRATIVEQA_FILE, narrativeqa_data={}):
		with open(NARRATIVEQA_FILE, 'r', encoding='utf8', errors='ignore') as fp:
			fp.readline()   # skip header
			for row in csv.reader(fp):
				context, question, references = row[1], row[2], [row[3], row[4]]
				if context not in narrativeqa_data:
					narrativeqa_data[context] = {}
				
				if question not in narrativeqa_data[context]:
					narrativeqa_data[context][question] = {'references': references, 'candidates': set()}

			return narrativeqa_data

	narrativeqa_data = load_file(NARRATIVEQA_DEV_FILE)
	narrativeqa_data = load_file(NARRATIVEQA_TEST_FILE, narrativeqa_data)

	return narrativeqa_data

def load_mhpg_predictions(file):
	lines = []
	for line in Reader(open(file)):
		context = line['raw_summary'].replace('\n', ' ').strip()
		question = line['raw_ques'].strip()
		candidate = line['pred'].replace(" n't", "n't")

		if 'UNK' not in candidate:
			lines.append((context, question, candidate))
	return lines

def write_data_to_label(data_dict):
	samples = []

	# First converts the dictionary to entries in a list
	for context in data_dict:
		for question in data_dict[context]:
			references = data_dict[context][question]['references']
			candidates = data_dict[context][question]['candidates']

			if not are_two_answers_the_same(question, references[0], references[1]):
				candidates.add(references[1])
				references = [references[0]]

			candidates = prune_candidates(references, data_dict[context][question]['candidates'])
			reference = random.choice(references)

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
	with open('merge_predictions/to_label/narrativeqa.csv', 'w') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['context', 'question', 'reference', 'candidate', 'id'])
		for line in samples:
			writer.writerow(line)

def main():
	MHPG_PREDICTIONS_DIR = '/home/tony/CommonSenseMultiHopQA/out/nqa_baseline'
	MHPG_PREDICTIONS_FILE = [join(MHPG_PREDICTIONS_DIR, 'narrative_qa_valid.jsonl.merged1'), 
							 join(MHPG_PREDICTIONS_DIR, 'narrative_qa_valid.jsonl.merged2'),
							 join(MHPG_PREDICTIONS_DIR, 'narrative_qa_test.jsonl.merged1'),
							 join(MHPG_PREDICTIONS_DIR, 'narrative_qa_test.jsonl.merged2')]

	narrativeqa_data = load_narrativeqa_data()

	for f in MHPG_PREDICTIONS_FILE:
		for context, question, candidate in load_mhpg_predictions(f):
			if context in narrativeqa_data and question in narrativeqa_data[context]:
				narrativeqa_data[context][question]['candidates'].add(candidate)

	write_data_to_label(narrativeqa_data)

if __name__ == '__main__':
	main()