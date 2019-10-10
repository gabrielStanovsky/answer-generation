import csv
import gpt_2_simple as gpt2
import os
from os.path import isfile, join
from tqdm import tqdm 
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="1"

K = [1, 10, 100]
N_SAMPLES = 3

def get_generated_answer(prefix, output):
	""" GPT2 generation returns the prefix along with the generated answer 
	This function strips away the prefix from the generated output """

	output = [o[len(prefix):] for o in output]
	return output

def generate_samples_for_file(data_file, run_name, checkpoint_dir, length):
	csv_filename = join(checkpoint_dir, run_name, data_file.split('/')[-1] + '_generation')

	if isfile(csv_filename):
		return

	csvfile = open(csv_filename, 'w')
	writer = csv.writer(csvfile)

	tf.reset_default_graph()
	sess = gpt2.start_tf_sess()
	gpt2.load_gpt2(sess, run_name=run_name, checkpoint_dir=checkpoint_dir)

	with open(data_file) as f:
		# Skip header line
		f.readline()
		reader = csv.reader(f)

		# Iterate through lines of the data file
		for i, line in enumerate(tqdm(reader)):
			# Every 10 iterations, reset the graph since it gets slower over time
			if i % 10 == 0 and i > 0:
				tf.reset_default_graph()
				sess.close()
				sess = gpt2.start_tf_sess()
				gpt2.load_gpt2(sess, run_name=run_name, checkpoint_dir=checkpoint_dir)

			gpt2_input = line[0].strip()

			generated_answers = set()
			# Iterate through the different values of top_k
			for k in K:
				if k == 1: n_samples = 1
				else: n_samples = N_SAMPLES

				gpt2_output = gpt2.generate(sess, 
								 	 	    checkpoint_dir=checkpoint_dir, 
								 	 	    run_name=run_name, 
								 	 	    nsamples=n_samples, 
								 	 	    batch_size=n_samples, 
								 	 	    top_k=k, 
								 	 	    temperature=1,
								 	 	    prefix=gpt2_input,
								 	 	    truncate="<|endoftext|>",
								 	 	    return_as_list=True,
								 	 	    length=length)
				gpt2_output = get_generated_answer(gpt2_input, gpt2_output)
				generated_answers.update(gpt2_output)
			line += generated_answers
			writer.writerow(line)
		csvfile.close()

def generate_gpt2_on_cosmosqa():
	generate_samples_for_file('data/cosmosqa/dev.csv', 'cosmosqa', 'gpt_models', length=25)

def generate_gpt2_on_mcscript():
	generate_samples_for_file('data/mcscript/dev.csv', 'mcscript', 'gpt_models', length=15)
	generate_samples_for_file('data/mcscript/test.csv', 'mcscript', 'gpt_models', length=15)

def generate_gpt2_on_socialiqa():
	generate_samples_for_file('data/socialiqa/dev.csv', 'socialiqa', 'gpt_models', length=15)
	generate_samples_for_file('data/socialiqa/test.csv', 'socialiqa', 'gpt_models', length=15)

if __name__ == '__main__':
	generate_gpt2_on_cosmosqa()
	generate_gpt2_on_mcscript()
	generate_gpt2_on_socialiqa()
