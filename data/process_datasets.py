"""
Contains functions for processing each dataset in its raw form.

Each multiple-choice dataset is converted from their raw format into a CSV file for use in GPT2 finetuning and generation.
"""
import csv
from jsonlines import Reader
import os
import random
import spacy
from tqdm import tqdm
import xmltodict

random.seed(0)

def process_cosmosqa(RAW_FILE, OUTPUT_FILE):
	csvfile = open(OUTPUT_FILE, 'w')
	writer = csv.writer(csvfile)
	writer.writerow(['gpt_input', 'context', 'question', 'answer'])

	answer_writer = open(OUTPUT_FILE + '_answers', 'w')
	question_answer_writer = open(OUTPUT_FILE + '_question_answers', 'w')
	context_writer = open(OUTPUT_FILE + '_context', 'w')

	with open(RAW_FILE) as f:
		# Skip header line
		f.readline()

		reader = csv.reader(f)
		for line in reader:
			context = line[1].strip()
			question = line[2].strip()
			correct_answer_id = int(line[-1]) - 5
			correct_answer = line[correct_answer_id].strip()

			if correct_answer == 'None of the above choices .':
				continue

			# If we're doing training, then include the correct answer so it learns to predict it
			# Otherwise, leave the correct answer out so we can generate an answer
			gpt_input = context + ' ' + question
			if 'train.csv' in OUTPUT_FILE:
				gpt_input += ' ' +  correct_answer
				
			writer.writerow([gpt_input, context, question, correct_answer])
			answer_writer.write(correct_answer + '\n')
			question_answer_writer.write(question.__repr__() + '\n' + correct_answer.__repr__() + '\n')
			context_writer.write(context.__repr__() + '\n')

	csvfile.close()
	answer_writer.close()
	question_answer_writer.close()
	context_writer.close()

def process_mcscript(RAW_FILE, OUTPUT_FILE):
	csvfile = open(OUTPUT_FILE, 'w')
	writer = csv.writer(csvfile)
	writer.writerow(['gpt_input', 'context', 'question', 'answer'])
	
	answer_writer = open(OUTPUT_FILE + '_answers', 'w')
	question_answer_writer = open(OUTPUT_FILE + '_question_answers', 'w')
	context_writer = open(OUTPUT_FILE + '_context', 'w')
	
	# Iterate through XML structure of the raw file
	with open(RAW_FILE) as f:
		raw_data = xmltodict.parse(''.join(f.readlines()))
		for instance in raw_data['data']['instance']:
			if instance['questions'] == None:
				continue

			context = instance['text'].strip()
			question_dicts = instance['questions']['question']
			question_dicts = question_dicts if type(question_dicts) == list else [question_dicts] 

			# Iterate through the question dictionaries of the context
			for question_id, question_dict in enumerate(question_dicts):
				question = question_dict['@text'].strip()
				# Grab the right answer from the answer dictionaries
				for answer_dict in question_dict['answer']:
					if answer_dict['@correct'] == "True":
						correct_answer = answer_dict['@text'].strip()

						gpt_input = context + ' ' + question
						if 'train.csv' in OUTPUT_FILE:
							gpt_input += ' ' +  correct_answer
							
						writer.writerow([gpt_input, context, question, correct_answer])
						answer_writer.write(correct_answer + '\n')
						question_answer_writer.write(question.__repr__() + '\n' + correct_answer.__repr__() + '\n')
						context_writer.write(context.__repr__() + '\n')

	csvfile.close()
	answer_writer.close()
	question_answer_writer.close()
	context_writer.close()

def process_narrativeqa(OUTPUT_FILE, data_type):
	csvfile = open(OUTPUT_FILE, 'w')
	writer = csv.writer(csvfile)
	writer.writerow(['gpt_input', 'context', 'question', 'answer1', 'answer2'])
	
	answer_writer = open(OUTPUT_FILE + '_answers', 'w')
	question_answer_writer = open(OUTPUT_FILE + '_question_answers', 'w')
	context_writer = open(OUTPUT_FILE + '_context', 'w')

	# Contexts are stored in a separate file, so load them first
	SUMMARIES_FILE = 'raw_data/narrativeqa/third_party/wikipedia/summaries.csv'
	def load_summaries():
		summaries = {}
		with open(SUMMARIES_FILE, 'r') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True)
			next(spamreader) # skip header line

			for row in spamreader:
				if row[1] != data_type:
					continue
				doc_num = row[0]
				summaries[doc_num] = row[2].replace('\n', ' ')
		return summaries
	summaries = load_summaries()

	# Iterate through raw_data
	RAW_FILE = 'raw_data/narrativeqa/qaps.csv'
	with open(RAW_FILE, 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True)
		next(spamreader)
		
		for row in tqdm(spamreader):
			if row[1] != data_type:
				continue
			doc_num = row[0]
			context = summaries[doc_num].strip()

			# Skip contexts that are very long
			if len(context.split()) > 500:
				continue

			question = row[2].strip()
			correct_answers = [row[3].strip(), row[4].strip()]

			# If we're doing training, then sample one of the correct answers so it learns to predict it
			# Otherwise, leave the correct answer out so we can generate an answer
			gpt_input = context + ' ' + question
			if 'train.csv' in OUTPUT_FILE:
				gpt_input += ' ' +  random.choice(correct_answers)
			
			writer.writerow([gpt_input, context, question] + correct_answers)
			# Use the first reference as the "correct" answer
			answer_writer.write(correct_answers[0] + '\n')
			question_answer_writer.write(question.__repr__() + '\n' + correct_answers[0].__repr__() + '\n')
			context_writer.write(context.__repr__() + '\n')


	csvfile.close()
	answer_writer.close()
	question_answer_writer.close()
	context_writer.close()

def process_socialiqa(RAW_FILE, OUTPUT_FILE):
	csvfile = open(OUTPUT_FILE, 'w')
	writer = csv.writer(csvfile)
	writer.writerow(['gpt_input', 'context', 'question', 'answer'])

	answer_writer = open(OUTPUT_FILE + '_answers', 'w')
	question_answer_writer = open(OUTPUT_FILE + '_question_answers', 'w')
	context_writer = open(OUTPUT_FILE + '_context', 'w')

	for line in Reader(open(RAW_FILE)):
		context = line['context'].strip()
		question = line['question'].strip()
		correct_answer = line['answer' + line['correct']].strip()

		# If we're doing training, then include the correct answer so it learns to predict it
		# Otherwise, leave the correct answer out so we can generate an answer
		gpt_input = context + ' ' + question
		if 'train.csv' in OUTPUT_FILE:
			gpt_input += ' ' +  correct_answer
			
		writer.writerow([gpt_input, context, question, correct_answer])
		answer_writer.write(correct_answer + '\n')
		question_answer_writer.write(question.__repr__() + '\n' + correct_answer.__repr__() + '\n')
		context_writer.write(context.__repr__() + '\n')

	csvfile.close()
	answer_writer.close()
	question_answer_writer.close()
	context_writer.close()

if __name__ == '__main__':
	# COSMOSQA
	print('Processing COSMOSQA dataset...')
	if not os.path.isdir('data/cosmosqa'):
		os.makedirs('data/cosmosqa')
	process_cosmosqa('raw_data/cosmosqa/train.csv', 'data/cosmosqa/train.csv')
	process_cosmosqa('raw_data/cosmosqa/valid.csv', 'data/cosmosqa/dev.csv')

	# MCScript
	print('Processing MCScript dataset...')
	if not os.path.isdir('data/mcscript'):
		os.makedirs('data/mcscript')
	process_mcscript('raw_data/mcscript/train-data.xml', 'data/mcscript/train.csv')
	process_mcscript('raw_data/mcscript/dev-data.xml', 'data/mcscript/dev.csv')
	process_mcscript('raw_data/mcscript/test-data.xml', 'data/mcscript/test.csv')

	# NarrativeQA
	print('Processing NarrativeQA dataset...')
	if not os.path.isdir('data/narrativeqa'):
		os.makedirs('data/narrativeqa')
	process_narrativeqa('data/narrativeqa/train.csv', 'train')
	process_narrativeqa('data/narrativeqa/dev.csv', 'valid')
	process_narrativeqa('data/narrativeqa/test.csv', 'test')

	# SocialIQA
	print('Processing SocialIQA dataset...')
	if not os.path.isdir('data/socialiqa'):
		os.makedirs('data/socialiqa')
	process_socialiqa('raw_data/socialiqa/socialIQa_v1.4_trn.jsonl', 'data/socialiqa/train.csv')
	process_socialiqa('raw_data/socialiqa/socialIQa_v1.4_dev.jsonl', 'data/socialiqa/dev.csv')
	process_socialiqa('raw_data/socialiqa/socialIQa_v1.4_tst.jsonl', 'data/socialiqa/test.csv')