import gpt_2_simple as gpt2
from os.path import isdir, join

def train_gpt2_on_cosmosqa():
	checkpoint_dir='gpt_models'
	run_name = 'cosmosqa'

	if not isdir(join(checkpoint_dir, run_name)):
		sess = gpt2.start_tf_sess()
		gpt2.finetune(sess, model_name='124M', dataset='data/cosmosqa/train.csv', 
					  steps=30000, checkpoint_dir=checkpoint_dir, run_name=run_name, 
					  print_every=500, sample_every=500, save_every=2000, batch_size=1)
	else:
		print('Trained COSMOSQA directory already exists')

if __name__ == '__main__':
	train_gpt2_on_cosmosqa()
	# train_gpt2_on_mcscript()
	# train_gpt2_on_socialiqa()
