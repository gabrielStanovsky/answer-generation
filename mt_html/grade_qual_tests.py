import boto3
import json
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
	else:
		print('\tWorker', worker_id, 'rejected with accuracy', accuracy, '!')
		response = client.reject_qualification_request(QualificationRequestId=request_id)
	assert response['ResponseMetadata']['HTTPStatusCode'] == 200

def grade_qual_requests(qual_requests: list, answer_dict: dict):
	""" 
	Iterates through the qualification requests for a qualification type,
	scoring the answers for each request.

	Passes the qualification request if it has an accuracy >= `ACCURACY_THRESHOLD`.
	"""
	for request in qual_requests:
		guesses 	= xmltodict.parse(request['Answer'])['QuestionFormAnswers']['Answer']
		assert len(guesses) == len(answer_dict) # Check that all questions in test are answered

		# Compute accuracy for the current request
		choices 	= {g['QuestionIdentifier']: int(g['SelectionIdentifier']) for g in guesses}
		is_correct 	= {qid: (1 if choice in answer_dict[qid] else 0) for qid, choice in choices.items()}
		num_correct = sum(is_correct.values())
		accuracy 	= num_correct/len(guesses)

		# Approve or reject the request based on the accuracy
		accept_request(request['QualificationRequestId'], request['WorkerId'], accuracy)

def main():
	# First get a list of all the qualifications
	response = client.list_qualification_types(MustBeRequestable=True, MustBeOwnedByCaller=True)
	assert response['ResponseMetadata']['HTTPStatusCode'] == 200
	qual_dict = {d['Name']:d['QualificationTypeId'] for d in response['QualificationTypes']}

	# For each qualification, get all requests for that qual
	for qual_name, qual_id in qual_dict.items():
		assert qual_name in ANSWER_DICT

		qual_requests = get_qual_requests(qual_id)
		
		if len(qual_requests) > 0:
			print('Grading requests for ', qual_name.upper(), 'qual...')
			grade_qual_requests(qual_requests, ANSWER_DICT[qual_name])
			print()

if __name__ == '__main__':
	main()