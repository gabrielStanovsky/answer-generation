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
				 gpt2_model: str, 
				 vocab: Vocabulary = Vocabulary(),  
				 initializer: InitializerApplicator = InitializerApplicator()) -> None:
		super(GPT2ForQA, self).__init__(vocab)
		self.gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model)

		# binary cross entropy loss which takes in logits
		self.bce_loss = torch.nn.BCEWithLogitsLoss() 
		self.metrics  = {'accuracy': BooleanAccuracy()}

		initializer(self)

		# Log the number of trainable parameters in the model
		number_params = sum([numpy.prod(p.size()) for p in list(self.parameters()) if p.requires_grad])
		logger.info('Number of trainable model parameters: %d', number_params)

	@overrides
	def forward(self, 
				input: torch.Tensor, 					# input.size() 		  			= [batch_size, seq_len]
				answer_start_pos: torch.Tensor = None,  # answer_start_pos.size()   	= [batch_size]
				answer_end_pos: torch.Tensor = None,	# answer_end_pos.size()   		= [batch_size]
				metadata = None):
		bs, seq_len = input.size()
		return

	@overrides
	def get_metrics(self, reset: bool = False) -> Dict[str, float]:
		return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

if __name__ == '__main__':
	pass