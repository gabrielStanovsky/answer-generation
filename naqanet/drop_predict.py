import argparse
import json
from os.path import join
from pytorch_pretrained_bert import BertModel, BertTokenizer 
import torch
from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import move_to_device
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.ERROR)

def get_predictions(model, serialization_dir, reader, device):
	""" Generates predictions from a trained model on a reader """    
	dev = reader.read('raw_data/drop/drop_dataset_dev.json')
	vocab = Vocabulary.from_files(join(serialization_dir, 'vocabulary'))
	iterator = BasicIterator(batch_size = 1)
	iterator.index_with(vocab)

	dev_iter = iterator(dev, num_epochs=1)
	dev_batches = [batch for batch in dev_iter]
	dev_batches = move_to_device(dev_batches, device)
	
	predictions = {}
	with torch.no_grad():
		for batch in tqdm(dev_batches):
			out = model(**batch)
			assert len(out['question_id']) == 1
			assert len(out['answer']) == 1

			query_id = out['question_id'][0]
			
			if 'value' in out['answer'][0]:
				prediction = out['answer'][0]['value']
			elif 'count' in out['answer'][0]:
				prediction = out['answer'][0]['count'].item()
			else:
				raise ValueError()
			predictions[query_id] = prediction
	print(model.get_metrics())
	torch.cuda.empty_cache()
	return predictions

def main(weights_file, device):
	serialization_dir = serialization_dir = '/'.join(weights_file.split('/')[:-1])
	config 	= Params.from_file(join(serialization_dir, 'config.json'))
	reader 	= DatasetReader.from_params(config['validation_dataset_reader'])
	model 	= Model.load(config=config, serialization_dir=serialization_dir, 
						 weights_file=weights_file, cuda_device=device).eval()
	
	predictions = get_predictions(model, serialization_dir, reader, device)

	# Write out predictions to file
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