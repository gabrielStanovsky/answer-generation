import boto3
import json
from jsonlines import Reader
from os.path import isfile
from pprint import pprint
import xmltodict

config = json.load(open('/home/tony/.aws/mycredentials'))
endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
client = boto3.client('mturk', endpoint_url=endpoint_url, 
					  region_name=config['region_name'],
					  aws_access_key_id=config['aws_access_key_id'],
					  aws_secret_access_key=config['aws_secret_access_key'])

ACCURACY_THRESHOLD = 0.7 # If turkers get >= than 70% of quals right, accept them

# Answer key for the quals. 
ANSWER_DICT = {'cosmosqa': {}, 
			   'drop': {},
			   'mcscript': {},
			   'narrativeqa': {'1': [2, 3], '2': [3, 4],
							   '3': [4], 	'4': [5],
							   '5': [1], 	'6': [5],
							   '7': [1, 2], '8': [3],
							   '9': [2, 3, 4], '10': [5]},
			   'quoref': {},
			   'ropes': {},
			   'socialiqa': {}}

# File to store worker choices, scores, and whether they passed or not
WORKER_OUTPUT_FILE = 'worker_scores.jsonl'

def get_qual_types():
	response = client.list_qualification_types(MustBeRequestable=True, MustBeOwnedByCaller=True)
	assert response['ResponseMetadata']['HTTPStatusCode'] == 200
	return response['QualificationTypes']

def get_qual_requests(qual_id: str):
	response = client.list_qualification_requests(QualificationTypeId=qual_id)
	assert response['ResponseMetadata']['HTTPStatusCode'] == 200
	return response['QualificationRequests']

def accept_request(request_id: str, worker_id: str, accuracy: float):
	if accuracy >= ACCURACY_THRESHOLD:
		print('\tWorker', worker_id, 'passed with accuracy', accuracy, '!')
		response = client.accept_qualification_request(QualificationRequestId=request_id)
		assert response['ResponseMetadata']['HTTPStatusCode'] == 200
		return True
	else:
		print('\tWorker', worker_id, 'rejected with accuracy', accuracy, '!')
		response = client.reject_qualification_request(QualificationRequestId=request_id)
		assert response['ResponseMetadata']['HTTPStatusCode'] == 200
		return False

def grade_qual_requests(qual_name: str, qual_id: str, qual_requests: list, writer, seen: set):
	""" 
	Iterates through the qualification requests for a qualification type,
	scoring the answers for each request.

	Passes the qualification request if it has an accuracy >= `ACCURACY_THRESHOLD`.
	"""
	current_qual_answer_dict = ANSWER_DICT[qual_name].copy()

	for request in qual_requests:
		request_id 	= request['QualificationRequestId']
		worker_id 	= request['WorkerId']
		guesses 	= xmltodict.parse(request['Answer'])['QuestionFormAnswers']['Answer']
		assert len(guesses) == len(current_qual_answer_dict) # Check that all questions in test are answered

		# Compute accuracy for the current request
		choices 	= {g['QuestionIdentifier']: int(g['SelectionIdentifier']) for g in guesses}
		is_correct 	= {qid: (1 if choice in current_qual_answer_dict[qid] else 0) for qid, choice in choices.items()}
		num_correct = sum(is_correct.values())
		accuracy 	= num_correct/len(guesses)

		if (worker_id, qual_id) not in seen:
			# Approve or reject the request based on the accuracy
			accepted = accept_request(request_id, worker_id, accuracy)

			output_dict = {'worker_id': worker_id,
						   'request_id': request_id,
						   'qual_name': qual_name,
						   'qual_id': qual_id,
						   'submit_time': str(request['SubmitTime']),
						   'choices': choices,
						   'choices_correctness': is_correct,
						   'accuracy': accuracy,
						   'accepted': accepted}

			writer.write(json.dumps(output_dict) + '\n')
			
		else:
			print('\tAlready seen worker id:', worker_id, 'with id', qual_id)

def main():
	# Get list of the seen worker ids with the qual ids so we don't accidentally allow workers to take test multiple times
	seen = set()
	if isfile(WORKER_OUTPUT_FILE):
		for line in Reader(open(WORKER_OUTPUT_FILE)):
			seen.add((line['worker_id'], line['qual_id']))

	with open(WORKER_OUTPUT_FILE, 'a') as writer:
		# Get a list of all the qualifications
		qualifications = get_qual_types()
		# A dictionary of qualification names : qualification ids
		qual_dict = {d['Name']:d['QualificationTypeId'] for d in qualifications}

		# For each qualification, get all requests for that qual
		for qual_name, qual_id in qual_dict.items():
			assert qual_name in ANSWER_DICT
			qual_requests = get_qual_requests(qual_id)
			
			if len(qual_requests) > 0:
				print('Grading requests for ', qual_name.upper(), 'qual...')
				grade_qual_requests(qual_name, qual_id, qual_requests, writer, seen)

if __name__ == '__main__':
	main()