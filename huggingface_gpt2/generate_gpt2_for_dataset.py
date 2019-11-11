from allennlp.models.archival import load_archive
import csv
import numpy
import random
import torch
import os
from os.path import isfile, join
from tqdm import tqdm 
from transformers import GPT2LMHeadModel, GPT2Tokenizer

numpy.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

os.environ["CUDA_VISIBLE_DEVICES"]="1"		

K = [1, 10, 100]
NUM_SAMPLES = 2
context_sep = '----------------------------------------------------------------'
question_sep = '================================================================='

def generate_samples_for_file(input_file, model_archive):
	from generate_gpt2 import sample_sequence
	from dataset_reader import GPT2ForQADatasetReader
	from model import GPT2ForQA

	tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	archive = load_archive(model_archive)
	model = archive.model.gpt2_model
	model.to(0).eval()

	csv_filename = join('/'.join(model_archive.split('/')[:-1]), input_file.split('/')[-1] + '_generation')
	if isfile(csv_filename): # If generation file exists, skip.
		return
	csvfile = open(csv_filename, 'w')
	writer = csv.writer(csvfile)

	with open(input_file) as f:
		# Skip header line
		f.readline()
		for line in tqdm(csv.reader(f)):
			context, question = line[1], line[2]
			prompt = tokenizer.eos_token + ' ' + context + ' ' + context_sep + ' ' + question + ' ' + question_sep
			prompt_ids = tokenizer.encode(prompt)
			generated_answers = set()

			if len(prompt_ids) > 1000:
				continue

			# Iterate through the different values of top_k
			for k in K:
				num_samples = 1 if k == 1 else NUM_SAMPLES
				
				out = sample_sequence(model=model, context=prompt_ids, num_samples=num_samples,
									  length=20, top_k=k, device=0)
				# Remove prompt IDS from output
				out = out[:, len(prompt_ids):].tolist()
				for o in out:
					text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
					text = text[: text.find(tokenizer.eos_token)]
					generated_answers.add(text)
			
			line += generated_answers
			writer.writerow(line)
	csvfile.close()

if __name__ == '__main__':
	generate_samples_for_file('data/narrativeqa/dev.csv', 'huggingface_gpt2/models/narrativeqa_sep/model.tar.gz')
	generate_samples_for_file('data/narrativeqa/test.csv', 'huggingface_gpt2/models/narrativeqa_sep/model.tar.gz')