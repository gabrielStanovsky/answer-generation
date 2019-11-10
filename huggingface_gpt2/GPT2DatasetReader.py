import csv
from jsonlines import Reader
from json import dumps
import logging
import numpy as np
from overrides import overrides
from transformers import GPT2Tokenizer
from transformers.tokenization_gpt2 import PRETRAINED_VOCAB_FILES_MAP

from allennlp.common import Params, Tqdm
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, LabelField, ListField, TextField
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.data.instance import Instance

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("gpt2forqa")
class GPT2ForQADatasetReader(DatasetReader):
	def __init__(self, 
				 gpt2_model = 'gpt2',
				 lazy: bool = False) -> None:
		super().__init__(lazy)
		self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)

	@overrides
	def _read(self, file_path: str):
		with open(file_path) as f:
			# Skip header line
			f.readline()
			for line in csv.reader(f):
				context, question, answer = line[1], line[2], line[3]

				yield self.text_to_instance(context, question, answer)

	@overrides
	def text_to_instance(self, context, question, answer=None) -> Instance:
		context_tokens 	= self.tokenizer.tokenize(context)
		question_tokens = self.tokenizer.tokenize(question)
		
		input_tokens = context_tokens + question_tokens
		
		metadata = {'context': context, 
					'question': question,
					'context_tokens': context_tokens,
					'question_tokens': question_tokens, 
					'input_tokens': input_tokens}

		if answer:
			answer_start_pos = len(input_tokens)
			answer_tokens = self.tokenizer.tokenize(answer)
			input_tokens += answer_tokens + [self.tokenizer.eos_token]
			answer_end_pos = len(input_tokens)

			metadata['answer'] 				= answer
			metadata['answer_tokens'] 		= answer_tokens
			metadata['answer_start_pos'] 	= answer_start_pos
			metadata['answer_end_pos'] 		= answer_end_pos

		input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

		fields = {'input_ids': ArrayField(np.array(input_ids), dtype=np.int64),
				  'metadata': MetadataField(metadata)}

		if answer:
			fields['answer_start_pos'] 	= ArrayField(np.array(answer_start_pos), dtype=np.int64)
			fields['answer_end_pos'] 	= ArrayField(np.array(answer_end_pos), dtype=np.int64)

		return Instance(fields)

if __name__ == '__main__':
	reader = GPT2ForQADatasetReader()
	eos_token = reader.tokenizer.eos_token

	for instance in Tqdm.tqdm(reader._read('/home/tony/answer-generation/data/narrativeqa/train.csv')):
		input_ids = instance.fields['input_ids'].array
		input_tokens = instance.fields['metadata'].metadata['input_tokens']
		context = instance.fields['metadata'].metadata['context']
		context_tokens = instance.fields['metadata'].metadata['context_tokens']
		question = instance.fields['metadata'].metadata['question']
		question_tokens = instance.fields['metadata'].metadata['question_tokens']
		answer = instance.fields['metadata'].metadata['answer']
		answer_tokens = instance.fields['metadata'].metadata['answer_tokens']
		answer_start_pos = instance.fields['answer_start_pos'].array.item()
		answer_end_pos = instance.fields['answer_end_pos'].array.item()

		# Check tokenization
		assert context == reader.tokenizer.convert_tokens_to_string(context_tokens)
		assert question == reader.tokenizer.convert_tokens_to_string(question_tokens)
		assert answer == reader.tokenizer.convert_tokens_to_string(answer_tokens)
		assert input_tokens == context_tokens + question_tokens + answer_tokens + [eos_token] 

		# Check tokens to ids
		assert input_tokens == reader.tokenizer.convert_ids_to_tokens(input_ids) 

		# Check answer boundaries
		assert input_tokens[answer_start_pos:answer_end_pos] == answer_tokens + [eos_token]
