import csv
import hashlib
from itertools import chain
from os.path import join
import random
import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser','ner', 'tagger'])
INPUT_DATA_DIR = 'merge_predictions/merged_datasets'
OUTPUT_DATA_DIR = 'merge_predictions/sampled_predictions'
QUESTIONS_PER_HIT = 10

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
	print('Found', num_questions, 'candidate answers.', num_unique_questions, 'unique questions. Skipped', num_skipped, 'questions')
	return data

def write_data(data, output_fn):
	outfile = open(output_fn, 'w', encoding='utf-8')
	writer = csv.writer(outfile)
	num_output_lines = 0

	# Write the header row
	header_row = ['id']
	for i in range(1, QUESTIONS_PER_HIT+1):
		header_row += ['context'+str(i), 'question'+str(i), 'reference'+str(i), 'candidate'+str(i), 'source'+str(i), 'id'+str(i)]
	writer.writerow(header_row)
	assert len(header_row) == 6*QUESTIONS_PER_HIT + 1

	# Iterate through contexts, squashing nested dictionaries into a
	# flat dictionary and writing them out to CSV file
	for context in data:
		questions = []
		for question in data[context]:
			reference = data[context][question]['reference']
			for source in data[context][question]['candidates']:
				for candidate_dict in data[context][question]['candidates'][source]:
					questions.append({'question': question,
									  'reference': reference,
									  'candidate': candidate_dict['candidate'],
									  'source': source,
									  'hash_id': candidate_dict['hash_id']})
		num_output_lines += write_rows(writer, context, questions)
	outfile.close()
	print('Wrote out', num_output_lines, 'lines...\n')
	
def write_rows(writer, context, questions):
	# For each context, calculate the number of hits needed (getting rid of remainders)
	num_hits_needed = len(questions)//QUESTIONS_PER_HIT
	random.shuffle(questions)

	for i in range(0, num_hits_needed):
		current_questions = questions[i*QUESTIONS_PER_HIT:(i+1)*QUESTIONS_PER_HIT]

		# Turn list of dictionaries into a list of the dictionary values
		current_questions = list(chain(*[[context, q['question'], q['reference'], q['candidate'], q['source'], q['hash_id']] for q in current_questions]))
		assert len(current_questions) == 6*QUESTIONS_PER_HIT

		# MD5 hash of the current questions
		hash_object = hashlib.md5(current_questions.__repr__().encode())
		row_id = hash_object.hexdigest()

		row = [row_id] + current_questions
		writer.writerow(row)

	# Return the number of questions we wrote out
	return num_hits_needed*QUESTIONS_PER_HIT

def check_sampled_data(input_file, output_file, questions_per_hit):
	# Check that each sampled question was present in the input file.
	# Also check that we haven't written out a sampled question twice.
	# This checks that we wrote out our sampled lines correctly
	input_lines = set()
	for line in csv.reader(open(input_file)):
		input_lines.add(line.__repr__())

	seen_lines = set()
	with open(output_file) as f:
		header = f.readline().strip().split(',')
		assert len(header) == questions_per_hit*6+1
		for line in csv.reader(f):
			row_id = line[0]
			# Start loop at 1 b/c first element is the row_id
			for i in range(1, questions_per_hit*6+1, 6):
				sampled_line = line[i:i+6]
				assert sampled_line.__repr__() in input_lines
				assert sampled_line.__repr__() not in seen_lines
				seen_lines.add(sampled_line.__repr__())

def sample_gpt2(reference, candidates, sample_size):
    best_score = 0
    best_candidate = None
    sampled_candidates = []
    
    tokenized_reference = set([token.lemma_ for token in nlp(reference.lower())])
    
    # Get highest overlap candidate
    for c in candidates:
        tokenized_c = set([token.lemma_ for token in nlp(c['candidate'].lower())])
        overlap = len(tokenized_c.intersection(tokenized_reference))
        
        if overlap > best_score:
            best_score = overlap
            best_candidate = c
            
    # Remove best candidate from candidates
    if best_candidate != None:
        sampled_candidates.append(best_candidate)
        candidates.remove(best_candidate)
    
    # Sample from the remaining candidates    
    sampled_candidates += random.sample(candidates, sample_size-len(sampled_candidates))
    
    return sampled_candidates

def sample_drop():
	random.seed(1)
	fn = 'drop.csv'
	input_fn = join(INPUT_DATA_DIR, fn)
	output_fn = join(OUTPUT_DATA_DIR, fn)
	data = load_data(input_fn)
	
	write_data(data, output_fn)
	check_sampled_data(input_fn, output_fn, QUESTIONS_PER_HIT)

def sample_quoref():
	random.seed(1)
	fn = 'quoref.csv'
	input_fn = join(INPUT_DATA_DIR, fn)
	output_fn = join(OUTPUT_DATA_DIR, fn)
	data = load_data(input_fn)
	
	write_data(data, output_fn)
	check_sampled_data(input_fn, output_fn, QUESTIONS_PER_HIT)
	
def sample_ropes():
	random.seed(1)
	fn = 'ropes.csv'
	input_fn = join(INPUT_DATA_DIR, fn)
	output_fn = join(OUTPUT_DATA_DIR, fn)
	data = load_data(input_fn)

	max_samples = 3

	# Sampling a subset of the candidate answers
	for context in data:
		for question in data[context]:
			bert = data[context][question]['candidates']['bert']
			data[context][question]['candidates']['bert'] = random.sample(bert, min(len(bert), max_samples))

	write_data(data, output_fn)
	check_sampled_data(input_fn, output_fn, QUESTIONS_PER_HIT)

def sample_mcscript():
	random.seed(1)
	fn = 'mcscript.csv'
	input_fn = join(INPUT_DATA_DIR, fn)
	output_fn = join(OUTPUT_DATA_DIR, fn)
	data = load_data(input_fn)
	max_gpt2 = 3

	# Sampling a subset of the candidate answers
	# Need to make sure to pad number of questions per context with backtranslations to ensure
	# that the number of questions per hit is divisible by `NUM_QUESTIONS_PER_HIT` 
	for context in data:
		cur_count = 0

		for question in data[context]:
			if 'mhpg' in data[context][question]['candidates']:
				if random.random() < 0.5:
					cur_count += 1
				else:
					del data[context][question]['candidates']['mhpg']

		for question in data[context]:
			reference = data[context][question]['reference']
			if 'gpt2' in data[context][question]['candidates']:
				gpt2 = data[context][question]['candidates']['gpt2']
				sample_size = min(len(gpt2), max_gpt2)
				data[context][question]['candidates']['gpt2'] = sample_gpt2(reference, gpt2, sample_size)
				cur_count += sample_size

		# Here, we sample backtranslations to ensure that we can pad our number of questions
		# We want to average at least one backtranslation per questions
		remainder = QUESTIONS_PER_HIT - cur_count%QUESTIONS_PER_HIT
		num_bt_to_sample = 0 if remainder == QUESTIONS_PER_HIT else remainder
		if num_bt_to_sample > len(data[context]) or num_bt_to_sample >= 5:
			num_bt_to_sample = 0

		# Sample backtranslations
		sampled_bts = []
		for _ in range(3):
			for question in data[context]:
				if len(sampled_bts) == num_bt_to_sample:
					break

				if 'backtranslation' in data[context][question]['candidates']:
					bt = random.choice(data[context][question]['candidates']['backtranslation'])
					data[context][question]['candidates']['backtranslation'].remove(bt)
					sampled_bts.append((question, bt))

					if len(data[context][question]['candidates']['backtranslation']) == 0:
						del data[context][question]['candidates']['backtranslation']

		# Delete the original backtranslations
		for question in data[context]:
			if 'backtranslation' in data[context][question]['candidates']:
				del data[context][question]['candidates']['backtranslation']

		# Add sampled backtranslations back into data
		for question, bt in sampled_bts:
			if 'backtranslation' not in data[context][question]['candidates']:
				data[context][question]['candidates']['backtranslation'] = []
			data[context][question]['candidates']['backtranslation'].append(bt)

	write_data(data, output_fn)
	check_sampled_data(input_fn, output_fn, QUESTIONS_PER_HIT)

def sample_narrativeqa():
	random.seed(1)
	fn = 'narrativeqa.csv'
	input_fn = join(INPUT_DATA_DIR, fn)
	output_fn = join(OUTPUT_DATA_DIR, fn)
	data = load_data(input_fn)
	max_gpt2 = 3

	# Sampling a subset of the candidate answers
	# Need to make sure to pad number of questions per context with backtranslations to ensure
	# that the number of questions per hit is divisible by `NUM_QUESTIONS_PER_HIT` 
	for context in data:
		cur_count = 0

		for question in data[context]:
			if 'narrativeqa' in data[context][question]['candidates']:
				cur_count += 1

		for question in data[context]:
			if 'mhpg' in data[context][question]['candidates']:
				mhpg = data[context][question]['candidates']['mhpg']
				data[context][question]['candidates']['mhpg'] = random.sample(mhpg, 1)
				cur_count += 1

		for question in data[context]:
			reference = data[context][question]['reference']
			if 'gpt2' in data[context][question]['candidates']:
				gpt2 = data[context][question]['candidates']['gpt2']
				sample_size = min(len(gpt2), max_gpt2)
				data[context][question]['candidates']['gpt2'] = sample_gpt2(reference, gpt2, sample_size)
				cur_count += sample_size

		# Here, we sample backtranslations to ensure that we can pad our number of questions
		# We want to average at least one backtranslation per questions
		remainder = QUESTIONS_PER_HIT - cur_count%QUESTIONS_PER_HIT
		num_bt_to_sample = remainder if remainder >= len(data[context]) else remainder + QUESTIONS_PER_HIT
		
		# Sample backtranslations
		sampled_bts = []
		for _ in range(3):
			for question in data[context]:
				if len(sampled_bts) == num_bt_to_sample:
					break

				if 'backtranslation' in data[context][question]['candidates']:
					bt = random.choice(data[context][question]['candidates']['backtranslation'])
					data[context][question]['candidates']['backtranslation'].remove(bt)
					sampled_bts.append((question, bt))

					if len(data[context][question]['candidates']['backtranslation']) == 0:
						del data[context][question]['candidates']['backtranslation']

		# If we don't have enough to sample but we have more than remainder, sample down to remainder
		if len(sampled_bts) < num_bt_to_sample and len(sampled_bts) > remainder:
			sampled_bts = random.sample(sampled_bts, remainder)

		# Delete the original backtranslations
		for question in data[context]:
			if 'backtranslation' in data[context][question]['candidates']:
				del data[context][question]['candidates']['backtranslation']

		# Add sampled backtranslations back into data
		for question, bt in sampled_bts:
			if 'backtranslation' not in data[context][question]['candidates']:
				data[context][question]['candidates']['backtranslation'] = []
			data[context][question]['candidates']['backtranslation'].append(bt)

	write_data(data, output_fn)
	check_sampled_data(input_fn, output_fn, QUESTIONS_PER_HIT)

################## COSMOSQA and SOCIALIQA have separate writing functions
################## because each line has multiple context file
def write_cosmosqa_socialiqa_data(data, output_fn, questions_per_hit):
	outfile = open(output_fn, 'w', encoding='utf-8')
	writer = csv.writer(outfile)
	num_output_lines = 0

	# Write the header row
	header_row = ['id']
	for i in range(1, questions_per_hit+1):
		header_row += ['context'+str(i), 'question'+str(i), 'reference'+str(i), 'candidate'+str(i), 'source'+str(i), 'id'+str(i)]
	writer.writerow(header_row)
	assert len(header_row) == 6*questions_per_hit + 1

	# Iterate through contexts, squashing nested dictionaries into a
	# flat dictionary and writing them out to CSV file
	entries = []
	for context in data:
		questions = []
		for question in data[context]:
			reference = data[context][question]['reference']
			for source in data[context][question]['candidates']:
				for candidate_dict in data[context][question]['candidates'][source]:
					entries.append({'context': context,
									'question': question,
									'reference': reference,
									'candidate': candidate_dict['candidate'],
									'source': source,
									'hash_id': candidate_dict['hash_id']})

					if len(entries) % questions_per_hit == 0:
						write_cosmosqa_socialiqa_rows(writer, entries, questions_per_hit)
						num_output_lines += questions_per_hit
						entries = []
	outfile.close()
	print('Wrote out', num_output_lines, 'lines...\n')

def write_cosmosqa_socialiqa_rows(writer, entries, questions_per_hit):
	# Turn list of dictionaries into a list of the dictionary values
	entries = list(chain(*[[q['context'], q['question'], q['reference'], q['candidate'], q['source'], q['hash_id']] for q in entries]))
	assert len(entries) == 6*questions_per_hit

	# MD5 hash of the current questions
	hash_object = hashlib.md5(entries.__repr__().encode())
	row_id = hash_object.hexdigest()

	row = [row_id] + entries
	writer.writerow(row)

def sample_cosmosqa_socialiqa_data(data, max_gpt2, max_bt):
	total = max_bt + max_gpt2
	
	# Sampling a subset of the candidate answers
	for context in data:
		for question in data[context]:
			reference = data[context][question]['reference']

			gpt2 = data[context][question]['candidates']['gpt2']
			if 'backtranslation' in data[context][question]['candidates']:
				bt = data[context][question]['candidates']['backtranslation']

				if len(gpt2) <= max_gpt2:
					data[context][question]['candidates']['backtranslation'] = random.sample(bt, min(len(bt), total-len(gpt2)))
				elif len(bt) <= max_bt:
					data[context][question]['candidates']['gpt2'] = sample_gpt2(reference, gpt2, total-len(bt))
				else:
					data[context][question]['candidates']['gpt2'] = sample_gpt2(reference, gpt2, max_gpt2)
					data[context][question]['candidates']['backtranslation'] = random.sample(bt, min(len(bt), max_bt))
			else:
				data[context][question]['candidates']['gpt2'] = sample_gpt2(reference, gpt2, total)

	return data

def sample_cosmosqa():
	random.seed(1)
	questions_per_hit = 20

	fn = 'cosmosqa.csv'
	input_fn = join(INPUT_DATA_DIR, fn)
	output_fn = join(OUTPUT_DATA_DIR, fn)
	data = load_data(input_fn)
	
	data = sample_cosmosqa_socialiqa_data(data, max_gpt2=3, max_bt=2)
	write_cosmosqa_socialiqa_data(data, output_fn, questions_per_hit)
	check_sampled_data(input_fn, output_fn, questions_per_hit)

def sample_socialiqa():
	random.seed(1)
	questions_per_hit = 24

	fn = 'socialiqa.csv'
	input_fn = join(INPUT_DATA_DIR, fn)
	output_fn = join(OUTPUT_DATA_DIR, fn)
	data = load_data(input_fn)

	data = sample_cosmosqa_socialiqa_data(data, max_gpt2=2, max_bt=1)
	write_cosmosqa_socialiqa_data(data, output_fn, questions_per_hit)
	check_sampled_data(input_fn, output_fn, questions_per_hit)

def main():
	print()
	sample_cosmosqa()
	sample_socialiqa()
	sample_mcscript()
	sample_narrativeqa()
	sample_drop()
	sample_quoref()
	sample_ropes()

if __name__ == '__main__':
	main()