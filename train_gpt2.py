import gpt_2_simple as gpt2
import os
from os.path import isdir, join
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def train_gpt2_on_cosmosqa():
	print('Training a model on COSMOSQA')

	checkpoint_dir='gpt_models'
	run_name = 'cosmosqa'

	if not isdir(join(checkpoint_dir, run_name)):
		sess = gpt2.start_tf_sess()
		gpt2.finetune(sess, model_name='124M', dataset='data/cosmosqa/train.csv', 
					  steps=15000, checkpoint_dir=checkpoint_dir, run_name=run_name, 
					  print_every=10, sample_every=1000, save_every=3000, batch_size=1, max_checkpoints=10)
	else:
		print('Trained COSMOSQA directory already exists')

def train_gpt2_on_mcscript():
	print('Training a model on MCScript')

	checkpoint_dir='gpt_models'
	run_name = 'mcscript'

	if not isdir(join(checkpoint_dir, run_name)):
		sess = gpt2.start_tf_sess()
		gpt2.finetune(sess, model_name='124M', dataset='data/mcscript/train.csv', 
					  steps=10000, checkpoint_dir=checkpoint_dir, run_name=run_name, 
					  print_every=10, sample_every=1000, save_every=3000, batch_size=1, max_checkpoints=10)
	else:
		print('Trained MCScript directory already exists')

def train_gpt2_on_narrativeqa():
	print('Training a model on NarrativeQA')

	checkpoint_dir='gpt_models'
	run_name = 'narrativeqa'

	if not isdir(join(checkpoint_dir, run_name)):
		sess = gpt2.start_tf_sess()
		gpt2.finetune(sess, model_name='124M', dataset='data/narrativeqa/train.csv', 
					  steps=10000, checkpoint_dir=checkpoint_dir, run_name=run_name, 
					  print_every=10, sample_every=1000, save_every=3000, batch_size=1, max_checkpoints=10)
	else:
		print('Trained NarrativeQA directory already exists')

def train_gpt2_on_socialiqa():
	print('Training a model on SocialIQA')

	checkpoint_dir='gpt_models'
	run_name = 'socialiqa'

	if not isdir(join(checkpoint_dir, run_name)):
		sess = gpt2.start_tf_sess()
		gpt2.finetune(sess, model_name='124M', dataset='data/socialiqa/train.csv', 
					  steps=15000, checkpoint_dir=checkpoint_dir, run_name=run_name, 
					  print_every=10, sample_every=1000, save_every=3000, batch_size=1, max_checkpoints=10)
	else:
		print('Trained SocialIQA directory already exists')

if __name__ == '__main__':
	train_gpt2_on_cosmosqa()
	tf.reset_default_graph()
	train_gpt2_on_mcscript()
	tf.reset_default_graph()
	train_gpt2_on_narrativeqa()
	tf.reset_default_graph()
	train_gpt2_on_socialiqa()
