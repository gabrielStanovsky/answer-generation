import logging
import numpy
from overrides import overrides
from transformers import GPT2LMHeadModel
import torch
from torch.nn import CrossEntropyLoss
from typing import Dict

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics.categorical_accuracy import  CategoricalAccuracy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
@Model.register("gpt2forqa")
class GPT2ForQA(Model):
	def __init__(self, 
				 gpt2_model: str = 'gpt2', 
				 vocab: Vocabulary = Vocabulary(),  
				 initializer: InitializerApplicator = InitializerApplicator()) -> None:
		super(GPT2ForQA, self).__init__(vocab)
		self.gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model)
		self.loss = CrossEntropyLoss(reduction='sum')
		self.metrics = {'accuracy': CategoricalAccuracy()}
		initializer(self)

		# Log the number of trainable parameters in the model
		number_params = sum([numpy.prod(p.size()) for p in list(self.parameters()) if p.requires_grad])
		logger.info('Number of trainable model parameters: %d', number_params)

	@overrides
	def forward(self, 
				input_ids: torch.Tensor, 				# input.size() 		  			= [batch_size, seq_len]
				answer_start_pos: torch.Tensor = None,  # answer_start_pos.size()   	= [batch_size]
				answer_end_pos: torch.Tensor = None,	# answer_end_pos.size()   		= [batch_size]
				metadata = None):
		batch_size = input_ids.size(0)
		# logits.size() = [batch_size, seq_len, vocab_size]
		logits = self.gpt2_model(input_ids=input_ids)[0].float()
		output_dict = {'logits': logits, 'metadata': metadata}

		# Iterate over questions, computing the loss over answer tokens
		assert type(answer_start_pos) == type(answer_end_pos)
		if type(answer_start_pos) != type(None):
			loss = 0
			num_answer_words = 0
			for i in range(batch_size):
				start, end = answer_start_pos[i], answer_end_pos[i]

				# answer logits are the positions before the label so go one back
				answer_logits = logits[i][start-1:end-1]
				# labels are the positions from start and end
				labels = input_ids[i][start:end]

				num_answer_words += labels.size(0)
				loss += self.loss(answer_logits, labels)
				self.metrics['accuracy'](answer_logits, labels)

			# Divide loss by num_answer_words
			output_dict['loss'] = loss/num_answer_words
		return output_dict

	@overrides
	def get_metrics(self, reset: bool = False) -> Dict[str, float]:
		return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

if __name__ == '__main__':
	torch.cuda.manual_seed(0)
	import random
	random.seed(0)
	from allennlp.data.iterators.basic_iterator import BasicIterator
	from GPT2DatasetReader import GPT2ForQADatasetReader

	model = GPT2ForQA()
	reader = GPT2ForQADatasetReader(lazy=True)
	train_data = reader.read('/home/tony/answer-generation/data/narrativeqa/train.csv')
	train_iterator = BasicIterator(batch_size=10)

	for batch in train_iterator(train_data):
		model(**batch)
		break