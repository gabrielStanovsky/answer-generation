import hashlib
from pytorch_pretrained_bert import BertTokenizer
import string
import spacy
import string

nlp = spacy.load('en_core_web_sm', disable=['parser','ner', 'tagger'])	
STOP_WORDS = ['a', 'an', 'the'] + string.punctuation
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def bert_tokenization_length(context, question, reference, candidate):
	context_len = len(tokenizer.tokenize(context))
	question_len = len(tokenizer.tokenize(question))
	candidate_len = len(tokenizer.tokenize(candidate))
	reference_len = len(tokenizer.tokenize(reference))

	return max(context_len + question_len + candidate_len, context_len + question_len + reference_len)

def check_data_and_return_hash(context, question, reference, candidate):
	assert type(context) == type(question) == type(reference) == type(candidate) == str

	if context == '' or question == '' or reference == '' or candidate == '':
		return None

	sample = context + question + reference + candidate
	hash_object = hashlib.md5(sample.encode())
	hash_id = hash_object.hexdigest()

	return hash_id

def clean_string(s):
	return s.replace('\n', ' ').strip()
	
def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		pass

	return False

def match(references, candidate):
    if type(references) == str:
        references = [references]

    for reference in references:
        # Test for numerical equivalence
        if is_number(reference) and is_number(candidate) and float(reference) == float(candidate):
            return True
        
        # Remove stop words in reference and candidate and lowercase
        tokenized_reference 	 = [token.text for token in nlp(reference.lower())]
        tokenized_candidate 	 = [token.text for token in nlp(candidate.lower())]
        tokenized_reference 	  = [token for token in tokenized_reference if token not in STOP_WORDS]
        tokenized_candidate 	  = [token for token in tokenized_candidate if token not in STOP_WORDS]

        stripped_reference = ''.join(tokenized_reference)
        stripped_candidate = ''.join(tokenized_candidate)

        # Test for string equivalence	
        if stripped_reference == stripped_candidate:
            return True

    return False

def prune_candidates(references, candidates):
	"""	
		Prune down the list of candidates by doing pairwise checks to make sure
		they aren't the same as the reference or other candidates
	"""
	unique_candidates = []
	
	# Skip candidates that are equivalent to the references or other candidates
	for c in sorted(candidates):
		if not match(references, c) and not match(unique_candidates, c):
			unique_candidates.append(c)

	return list(sorted(unique_candidates))

def prune_and_sort_samples(samples):
	""" Prunes the samples by filtering out multiple hash ids and 
		then sort the samples, first by context, then question, then candidates
	"""
	hash_set = set()
	pruned_samples = []
	
	for sample in samples:
		if sample[-1] in hash_set:
			continue
		else:
			hash_set.add(sample[-1])
			pruned_samples.append(sample)

	pruned_samples = sorted(pruned_samples, key = lambda x: x[3])
	pruned_samples = sorted(pruned_samples, key = lambda x: x[1])
	pruned_samples = sorted(pruned_samples, key = lambda x: x[0])

	return pruned_samples
