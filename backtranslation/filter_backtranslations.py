# Uses a trained discrminiative filtering model to filter out backtranslations
from allennlp.models.archival import load_archive
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.predictors import Predictor
import csv
from os.path import isfile
import sys
from tqdm import tqdm

sys.path.append('/home/tony/QAMetric/Discriminative-Filtering/files')
sys.path.append('.')

from model import Bert
from dataset_reader import ParaphraseDatasetReader
from merge_predictions.merge_utils import bert_tokenization_length, match

DEVICE = 0

def filter_narrativeqa():
	archive = load_archive('backtranslation/narrativeqa_discriminator/model.tar.gz')
	dataset_reader_params = archive.config.pop('dataset_reader')
	reader = DatasetReader.by_name(dataset_reader_params.pop('type')).from_params(dataset_reader_params)
	
	predictor = Predictor(archive.model.to(DEVICE).eval(), reader)

	def filter_narrativeqa_file(predictor, bt_file):
		if isfile(bt_file + '.filtered'):
			return

		output_file = open(bt_file + '.filtered', 'w')
		writer = csv.writer(output_file)

		with open(bt_file, 'r', encoding='utf8', errors='ignore') as fp:
			for line in tqdm(csv.reader(fp)):
				context, question, answer = line[1], line[2], line[3]
				backtranslations = [reader.clean_string(eval(c)[0]) for c in line[5:]]
				
				# Prune backtranslations from the original line
				line = line[:5]

				# Iterate through backtranslations, filtering out the ones that don't 
				# pass our discriminator threshold.
				for bt in backtranslations:
					if bert_tokenization_length(context, question, answer, bt) + 4 > 512 or \
					   match(answer, bt) or len(bt.split()) > (len(answer.split())+3)*3:
						continue
					
					instance = reader.text_to_instance(context + ' ' + question, bt)
					output_dict = predictor.predict_instance(instance)

					if output_dict['class_probabilities'] > 0.4:
						line.append(bt)

				writer.writerow(line)
		output_file.close()

	filter_narrativeqa_file(predictor, 'backtranslation/narrativeqa/dev.csv_answers.backtranslations')
	filter_narrativeqa_file(predictor, 'backtranslation/narrativeqa/test.csv_answers.backtranslations')

def filter_mcscript():
	pass

def filter_cosmosqa():
	pass

def filter_socialiqa():
	pass

def main():
	filter_narrativeqa()
	filter_mcscript()
	filter_cosmosqa()
	filter_socialiqa()
	
if __name__ == '__main__':
	main()