import csv
from jsonlines import Reader
from os.path import join
import spacy
from spacy.lang.en.stop_words import STOP_WORDS 
import string
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm', disable=['parser','ner', 'tagger'])	
STOP_WORDS.update(string.punctuation)

from merge_utils import bert_tokenization_length, clean_string, check_data_and_return_hash, prune_and_sort_samples, prune_candidates

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

	def load_file(NARRATIVEQA_FILE, data={}):
		with open(NARRATIVEQA_FILE, 'r', encoding='utf8', errors='ignore') as fp:
			fp.readline()   # skip header
			for row in csv.reader(fp):
				context = clean_string(row[1])
				question = clean_string(row[2])
				references = [clean_string(row[3]), clean_string(row[4])]

				if context not in data:
					data[context] = {}
				
				if question not in data[context]:
					data[context][question] = {'references': references, 'candidates': {}}

			return data

	data = load_file(NARRATIVEQA_DEV_FILE)
	data = load_file(NARRATIVEQA_TEST_FILE, data)

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
	
def load_mhpg_predictions(file):
	lines = []
	for line in Reader(open(file)):
		context = clean_string(line['raw_summary'])
		question = clean_string(line['raw_ques'].strip())
		candidate = clean_string(line['pred']).replace(" n't", "n't").replace(" 's", "'s")

		if 'UNK' not in candidate:
			lines.append((context, question, {candidate: 'mhpg'}))
			
	return lines

def load_backtranslations(file):
	lines = []
	with open(file, 'r', encoding='utf8', errors='ignore') as fp:
		for row in csv.reader(fp):
			context = clean_string(row[1])
			question = clean_string(row[2])

			# c[0] because c[1] is the score of the backtranslation
			candidates = {clean_string(c) : 'backtranslation' for c in row[5:]}
			lines.append((context, question, candidates))
	return lines

def write_data_to_label(data_dict):
	samples = []

	# First converts the dictionary to entries in a list
	for context in tqdm(data_dict):
		for question in data_dict[context]:
			references = data_dict[context][question]['references']
			candidates = data_dict[context][question]['candidates']

			if not are_two_answers_the_same(question, references[0], references[1]):
				candidates.update({references[1]:'narrativeqa'})
				references = [references[0]]

			unique_candidates_keys = prune_candidates(references, candidates)

			# Write out the first reference as the "gold" reference
			reference = references[0]

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
	with open('merge_predictions/merged_datasets/narrativeqa.csv', 'w') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['context', 'question', 'reference', 'candidate', 'source', 'id'])
		for line in samples:
			writer.writerow(line)

def main():
	# Paths to prediction files
	GPT_PREDICTIONS_DIR = '/home/tony/answer-generation/huggingface_gpt2/models/narrativeqa'
	GPT2_PREDICTIONS_FILE = [join(GPT_PREDICTIONS_DIR, 'dev.csv_generation'), join(GPT_PREDICTIONS_DIR, 'test.csv_generation')]

	MHPG_PREDICTIONS_DIR = '/home/tony/CommonSenseMultiHopQA/out/nqa_baseline'
	MHPG_PREDICTIONS_FILE = [join(MHPG_PREDICTIONS_DIR, 'narrative_qa_valid.jsonl.merged1'), 
							 join(MHPG_PREDICTIONS_DIR, 'narrative_qa_valid.jsonl.merged2'),
							 join(MHPG_PREDICTIONS_DIR, 'narrative_qa_test.jsonl.merged1'),
							 join(MHPG_PREDICTIONS_DIR, 'narrative_qa_test.jsonl.merged2')]

	BACKTRANSLATION_DIR = '/home/tony/answer-generation/backtranslation/narrativeqa'
	BACKTRANSLATION_FILES = [join(BACKTRANSLATION_DIR, 'dev.csv_answers.backtranslations.filtered'), 
							 join(BACKTRANSLATION_DIR, 'test.csv_answers.backtranslations.filtered')]

	data = load_narrativeqa_data()

	for f in GPT2_PREDICTIONS_FILE:
		for context, question, candidates in load_gpt2_predictions(f):
			data[context][question]['candidates'].update(candidates)

	for f in MHPG_PREDICTIONS_FILE:
		for context, question, candidate in load_mhpg_predictions(f):
			if context in data and question in data[context]:
				data[context][question]['candidates'].update(candidate)

	for f in BACKTRANSLATION_FILES:
		for context, question, candidate in load_backtranslations(f):
			if context in data and question in data[context]:
				data[context][question]['candidates'].update(candidate)

	write_data_to_label(data)

if __name__ == '__main__':
	main()