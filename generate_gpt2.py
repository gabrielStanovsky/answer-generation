import gpt_2_simple as gpt2
from os.path import isdir, join

def generate_gpt2_on_cosmosqa():
	checkpoint_dir='gpt_models'
	run_name = 'cosmosqa'

def generate_gpt2_on_mcscript():
	checkpoint_dir='gpt_models'
	run_name = 'mcscript'

def generate_gpt2_on_socialiqa():
	checkpoint_dir='gpt_models'
	run_name = 'socialiqa'

if __name__ == '__main__':
	generate_gpt2_on_cosmosqa()
	generate_gpt2_on_mcscript()
	generate_gpt2_on_socialiqa()
