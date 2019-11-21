import csv
import hashlib
from itertools import chain
from math import ceil
from os.path import join
import random

INPUT_DATA_DIR = 'merge_predictions/merged_datasets'
OUTPUT_DATA_DIR = 'merge_predictions/sampled_predictions'
MAX_QUESTIONS_PER_HIT = 20

def load_data(data_file):
	print('SAMPLING', data_file.split('/')[-1].upper())
	print('='*40)
	data = {}
	num_skipped = 0
	num_questions = 0
	num_unique_questions = 0
	with open(data_file) as f:
		# Skip header
		f.readline()
		reader = csv.reader(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True)
		for line in reader:
			context, question, reference, candidate, source, hash_id = line

			if context not in data:
				data[context] = {}
				
			if question not in data[context]:
				data[context][question] = {'reference': reference, 'candidates': {}}
				num_unique_questions += 1

			# We found that DROP has questions with multiple references so 
			# we just skip the duplicates and log how often they occur
			if data[context][question]['reference'] != reference:
				num_skipped += 1
				continue

			if source not in data[context][question]['candidates']:
				data[context][question]['candidates'][source] = []
			
			data[context][question]['candidates'][source].append({'candidate': candidate, 'hash_id': hash_id})
			num_questions += 1

	print('Found', num_questions, 'questions...', num_unique_questions, 'unique questions.... Skipped', num_skipped, 'questions')
	return data

def write_data(data, output_fn):
	outfile = open(output_fn, 'w')
	writer = csv.writer(outfile)
	num_output_lines = 0

	# Write the header row
	header_row = ['context', 'id']
	for i in range(1, MAX_QUESTIONS_PER_HIT+1):
		header_row += ['question'+str(i), 'reference'+str(i), 'candidate'+str(i), 'source'+str(i), 'id'+str(i)]
	
	writer.writerow(header_row)
	assert len(header_row) == 5*MAX_QUESTIONS_PER_HIT + 2

	# Iterate through contexts, getting questions and writing them out to CSV file
	for context in data:
		# Squash nested dictionaries into a flat dictionary
		questions = []
		for question in data[context]:
			reference = data[context][question]['reference']
			for source in data[context][question]['candidates']:
				for candidate_dict in data[context][question]['candidates'][source]:
					questions.append({'question': question,
									  'reference': reference,
									  'candidate': candidate_dict['candidate'],
									  'source': source,
									  'hash_id': candidate_dict['hash_id']
									})

		write_rows(writer, context, questions)
		num_output_lines += len(questions)

	outfile.close()
	print('Wrote out', num_output_lines, 'lines...\n')
	
def write_rows(writer, context, questions):
	# For each context, evenly split the number of questions per HIT and write to file
	num_hits_needed = ceil(len(questions)/MAX_QUESTIONS_PER_HIT)
	num_questions_per_hit = ceil(len(questions)/num_hits_needed)

	for i in range(0, len(questions), num_questions_per_hit):
		current_questions = questions[i:i+num_questions_per_hit]
		num_current_questions = len(current_questions)
		
		# Turn list of dictionaries into a list of dictionary values
		current_questions = list(chain(*[[q['question'], q['reference'], q['candidate'], q['source'], q['hash_id']] for q in current_questions]))
		assert len(current_questions)/5 == num_current_questions

		# MD5 hash of the current questions
		hash_object = hashlib.md5(current_questions.__repr__().encode())
		row_id = hash_object.hexdigest()

		row = [context, row_id] + current_questions + ['none']*max(0, (5*(MAX_QUESTIONS_PER_HIT-num_current_questions)))
		assert len(row) == 5*MAX_QUESTIONS_PER_HIT + 2

		writer.writerow(row)

def check_sampled_data(input_file, output_file):
	# Check that each sampled question was present in the input file.
	# Also check that we haven't written out a sampled question twice.
	# This checks that we wrote out our sampled lines correctly
	input_lines = set()
	for line in csv.reader(open(input_file)):
		input_lines.add(line.__repr__())

	seen_lines = set()
	with open(output_file) as f:
		header = f.readline().strip().split(',')
		assert len(header) == MAX_QUESTIONS_PER_HIT*5+2
		for line in csv.reader(f):
			context, row_id = line[0], line[1]
			for i in range(2, MAX_QUESTIONS_PER_HIT*5+2, 5):
				sampled_line = [context]+line[i:i+5]
				if sampled_line.__repr__() in input_lines:
					assert sampled_line.__repr__() not in seen_lines
					seen_lines.add(sampled_line.__repr__())
				else:
					assert sampled_line == [context] + ['none', 'none', 'none', 'none', 'none']

def sample_cosmosqa():
	random.seed(1)
	fn = 'cosmosqa.csv'
	input_fn = join(INPUT_DATA_DIR, fn)
	output_fn = join(OUTPUT_DATA_DIR, fn)
	data = load_data(input_fn)

	write_data(data, output_fn)
	check_sampled_data(input_fn, output_fn)

def sample_drop():
	random.seed(1)
	fn = 'drop.csv'
	input_fn = join(INPUT_DATA_DIR, fn)
	output_fn = join(OUTPUT_DATA_DIR, fn)
	data = load_data(input_fn)
	
	write_data(data, output_fn)

	check_sampled_data(input_fn, output_fn)

def sample_mcscript():
	random.seed(1)
	fn = 'mcscript.csv'
	input_fn = join(INPUT_DATA_DIR, fn)
	output_fn = join(OUTPUT_DATA_DIR, fn)
	data = load_data(input_fn)

	write_data(data, output_fn)
	check_sampled_data(input_fn, output_fn)

def sample_narrativeqa():
	random.seed(1)
	fn = 'narrativeqa.csv'
	input_fn = join(INPUT_DATA_DIR, fn)
	output_fn = join(OUTPUT_DATA_DIR, fn)
	data = load_data(input_fn)

	write_data(data, output_fn)
	check_sampled_data(input_fn, output_fn)

def sample_quoref():
	random.seed(1)
	fn = 'quoref.csv'
	input_fn = join(INPUT_DATA_DIR, fn)
	output_fn = join(OUTPUT_DATA_DIR, fn)
	data = load_data(input_fn)
	
	write_data(data, output_fn)
	check_sampled_data(input_fn, output_fn)
	
def sample_ropes():
	random.seed(1)
	fn = 'ropes.csv'
	input_fn = join(INPUT_DATA_DIR, fn)
	output_fn = join(OUTPUT_DATA_DIR, fn)
	data = load_data(input_fn)

	max_num_samples = [2, 3]

	# Sampling a subset of the candidate answers
	for context in data:
		for question in data[context]:
			bert = data[context][question]['candidates']['bert']
			data[context][question]['candidates']['bert'] = random.sample(bert, min(len(bert), random.choice(max_num_samples)))

	write_data(data, output_fn)
	check_sampled_data(input_fn, output_fn)

def sample_socialiqa():
	random.seed(1)
	fn = 'socialiqa.csv'
	input_fn = join(INPUT_DATA_DIR, fn)
	output_fn = join(OUTPUT_DATA_DIR, fn)
	data = load_data(input_fn)

	write_data(data, output_fn)
	check_sampled_data(input_fn, output_fn)

def main():
	print()
	# sample_cosmosqa()
	sample_drop()
	# sample_mcscript()
	# sample_narrativeqa()
	sample_quoref()
	sample_ropes()
	# sample_socialiqa()

if __name__ == '__main__':
	main()