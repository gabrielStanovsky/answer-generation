import argparse
import json
from os.path import join
from pytorch_pretrained_bert import BertModel, BertTokenizer 
import torch
from allennlp.models.model import Model
from allennlp.training.metrics.drop_em_and_f1 import DropEmAndF1
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import move_to_device
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.ERROR)

import sys
sys.path.append('.')

from nabert.augmented_bert_templated import NumericallyAugmentedBERTT
from nabert.drop_reader import BertDropTokenizer, BertDropReader, BertDropTokenIndexer

def get_predictions(abert, reader, device):
	""" Generates predictions from a trained model on a reader """    
	dev = reader.read('raw_data/drop/drop_dataset_dev.json')
	iterator = BasicIterator(batch_size = 1)
	iterator.index_with(Vocabulary())

	dev_iter = iterator(dev, num_epochs=1)
	dev_batches = [batch for batch in dev_iter]
	dev_batches = move_to_device(dev_batches, device)
	
	predictions = {}
	with torch.no_grad():
		for batch in tqdm(dev_batches):
			out = abert(**batch)
			assert len(out['question_id']) == 1
			assert len(out['answer']) == 1

			query_id = out['question_id'][0]
			prediction = out['answer'][0]['value']
			predictions[query_id] = prediction
	torch.cuda.empty_cache()
	return predictions

def main(weights_file, device):
	tokenizer 		= BertDropTokenizer('bert-base-uncased')
	token_indexer 	= BertDropTokenIndexer('bert-base-uncased')
	reader 			= BertDropReader(tokenizer, {'tokens': token_indexer}, extra_numbers=[100, 1], exp_search='template')

	abert 			= NumericallyAugmentedBERTT(Vocabulary(), 'bert-base-uncased', special_numbers=[100, 1])
	abert.load_state_dict(torch.load(weights_file, map_location='cpu'))
	abert.to(device).eval()
	
	predictions 	= get_predictions(abert, reader, device)

	# Write out predictions to file
	serialization_dir = '/'.join(weights_file.split('/')[:-1])
	predictions_file = weights_file.split('/')[-1].split('.')[0] + '_dev_pred.json'
	predictions_file = join(serialization_dir, predictions_file)

	with open(predictions_file, "w") as writer:
		writer.write(json.dumps(predictions, indent=4) + "\n")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--device', type=int, default=-0, required=False, help='device ID to use (-1 for CPU)')
	parser.add_argument('-w', '--weights_file', type=str, required=True, help='path to weights file to use')
	args = parser.parse_args()

	print(args)
	main(args.weights_file, args.device)