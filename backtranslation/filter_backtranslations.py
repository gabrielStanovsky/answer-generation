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

		for line in tqdm(csv.reader(open(bt_file, 'r', encoding='utf8', errors='ignore'))):
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
	archive = load_archive('backtranslation/mcscript_discriminator/model.tar.gz')
	dataset_reader_params = archive.config.pop('dataset_reader')
	reader = DatasetReader.by_name(dataset_reader_params.pop('type')).from_params(dataset_reader_params)
	predictor = Predictor(archive.model.to(DEVICE).eval(), reader)

	def filter_mcscript_file(predictor, bt_file):
		if isfile(bt_file + '.filtered'):
			return

		output_file = open(bt_file + '.filtered', 'w')
		writer = csv.writer(output_file)

		for line in tqdm(csv.reader(open(bt_file, 'r', encoding='utf8', errors='ignore'))):
			context, question, answer = line[1], line[2], line[3]
			backtranslations = [eval(c)[0] for c in line[5:]]

			scores = []
			for bt in backtranslations:
				if bert_tokenization_length(context, question, answer, bt) > 508 or \
					match(answer, bt) or len(bt.split()) > (len(answer.split())+3)*3:
					continue
					
				instance = reader.text_to_instance(context + ' ' + question, bt)
				output_dict = predictor.predict_instance(instance)
				scores.append((bt, output_dict['class_probabilities']))
			
			# Sort backtranslations by scores and filter them
			scores = sorted(scores, key=lambda x: x[1], reverse=True)
			# Take candidates that are over 0.3
			bts = [score[0] for score in scores if score[1] >= .3]
			# If nothing, then take the top 3
			if len(bts) == 0:
				bts = [score[0] for score in scores[:3]]

			# Prune out original backtranslations, and append the ones that made the cut
			line = line[:4]
			for bt in bts:
				line.append(bt)

			writer.writerow(line)
		output_file.close()

	filter_mcscript_file(predictor, 'backtranslation/mcscript/dev.csv_answers.backtranslations')
	filter_mcscript_file(predictor, 'backtranslation/mcscript/test.csv_answers.backtranslations')

def filter_cosmosqa():
	archive = load_archive('backtranslation/cosmosqa_discriminator/model.tar.gz')
	dataset_reader_params = archive.config.pop('dataset_reader')
	reader = DatasetReader.by_name(dataset_reader_params.pop('type')).from_params(dataset_reader_params)
	predictor = Predictor(archive.model.to(DEVICE).eval(), reader)

	bt_file = 'backtranslation/cosmosqa/dev.csv_answers.backtranslations'	
	if isfile(bt_file + '.filtered'):
		return

	output_file = open(bt_file + '.filtered', 'w')
	writer = csv.writer(output_file)

	for line in tqdm(csv.reader(open(bt_file, 'r', encoding='utf8', errors='ignore'))):
		context, question, answer = line[1], line[2], line[3]
		backtranslations = [eval(c)[0] for c in line[5:]]

		scores = []
		for bt in backtranslations:
			if bert_tokenization_length(context, question, answer, bt) > 508 or \
				match(answer, bt) or len(bt.split()) > (len(answer.split())+3)*3:
				continue
				
			instance = reader.text_to_instance(context + ' ' + question, bt)
			output_dict = predictor.predict_instance(instance)
			scores.append((bt, output_dict['class_probabilities']))
		
		# Sort backtranslations by scores and filter them
		scores = sorted(scores, key=lambda x: x[1], reverse=True)
		# Take candidates that are over 0.3
		bts = [score[0] for score in scores if score[1] >= .3]
		# If nothing, then take the top 3
		if len(bts) == 0:
			bts = [score[0] for score in scores[:3]]

		# Prune out original backtranslations, and append the ones that made the cut
		line = line[:4]
		for bt in bts:
			line.append(bt)

		writer.writerow(line)
	output_file.close()

def filter_socialiqa():
	archive = load_archive('backtranslation/socialiqa_discriminator/model.tar.gz')
	dataset_reader_params = archive.config.pop('dataset_reader')
	reader = DatasetReader.by_name(dataset_reader_params.pop('type')).from_params(dataset_reader_params)
	predictor = Predictor(archive.model.to(DEVICE).eval(), reader)

	def filter_socialiqa_file(predictor, bt_file):
		if isfile(bt_file + '.filtered'):
			return

		output_file = open(bt_file + '.filtered', 'w')
		writer = csv.writer(output_file)

		for line in tqdm(csv.reader(open(bt_file, 'r', encoding='utf8', errors='ignore'))):
			context, question, answer = line[1], line[2], line[3]
			backtranslations = [eval(c)[0] for c in line[5:]]

			scores = []
			for bt in backtranslations:
				if bert_tokenization_length(context, question, answer, bt) > 508 or \
					match(answer, bt) or len(bt.split()) > (len(answer.split())+3)*3:
					continue
					
				instance = reader.text_to_instance(context + ' ' + question, bt)
				output_dict = predictor.predict_instance(instance)
				scores.append((bt, output_dict['class_probabilities']))
			
			# Sort backtranslations by scores and filter them
			scores = sorted(scores, key=lambda x: x[1], reverse=True)
			# Take candidates that are over 0.3
			bts = [score[0] for score in scores if score[1] >= .3]
			# If nothing, then take the top 3
			if len(bts) == 0:
				bts = [score[0] for score in scores[:3]]

			# Prune out original backtranslations, and append the ones that made the cut
			line = line[:4]
			for bt in bts:
				line.append(bt)

			writer.writerow(line)
		output_file.close()

	filter_socialiqa_file(predictor, 'backtranslation/socialiqa/dev.csv_answers.backtranslations')
	filter_socialiqa_file(predictor, 'backtranslation/socialiqa/test.csv_answers.backtranslations')


def main():
	filter_narrativeqa()
	filter_mcscript()
	filter_cosmosqa()
	filter_socialiqa()
	
if __name__ == '__main__':
	main()