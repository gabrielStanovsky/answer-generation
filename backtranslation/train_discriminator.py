# Trains a discriminative model for filtering backtranslatins
# The model is trained to distinguish backtranslations vs genuine answers
import os

discriminator_filtering_dirname = '/home/tony/QAMetric/Discriminative-Filtering'

def main():
	# Train Narrativeqa Discriminator
	cmd = 'cd ' + discriminator_filtering_dirname + ' ;  \
		   allennlp train /home/tony/answer-generation/backtranslation/narrativeqa_config.json \
		   --include-package files -s /home/tony/answer-generation/backtranslation/narrativeqa_discriminator/'
	os.system(cmd)

	# Train MCScript Discriminator
	cmd = 'cd ' + discriminator_filtering_dirname + ' ;  \
		   allennlp train /home/tony/answer-generation/backtranslation/mcscript_config.json \
		   --include-package files -s /home/tony/answer-generation/backtranslation/mscript_discriminator/'
	os.system(cmd)

if __name__ == '__main__':
	main()