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
			   'narrativeqa': {'1': [2, 3], '2': [4, 5],
							   '3': [4], 	'4': [5],
							   '5': [1], 	'6': [5],
							   '7': [1, 2], '8': [3],
							   '9': [2, 3], '10': [5]},
			   'quoref': {},
			   'ropes': {},
			   'socialiqa': {}}

def get_qual_requests(qual_id):
	response = client.list_qualification_requests(QualificationTypeId=qual_id)
	assert response['ResponseMetadata']['HTTPStatusCode'] == 200
	return response['QualificationRequests']

def grade_qual_requests(qual_requests, answer_dict):
	""" 
	Iterates through the qualification requests for a qualification type,
	scoring the answers for each request.

	Passes the qualification request if it has an accuracy >= `ACCURACY_THRESHOLD`.
	"""
	for request in qual_requests:
		# Grab elements from the request
		request_id = request['QualificationRequestId']
		worker_id = request['WorkerId']
		guesses = xmltodict.parse(request['Answer'])['QuestionFormAnswers']['Answer']

		# Check that all questions in test are answered
		assert len(guesses) == len(answer_dict)

		# Compute accuracy for that request
		num_correct = len([1 for g in guesses if int(g['SelectionIdentifier']) in answer_dict[g['QuestionIdentifier']]])
		accuracy = num_correct/len(guesses)
		print(accuracy)

		# Approve or reject the request based on the accuracy
		if accuracy >= ACCURACY_THRESHOLD:
			print('worker', worker_id, 'passed!')
			client.accept_qualification_request(QualificationRequestId=request_id)
		else:
			print('worker', worker_id, 'rejected!')
			client.reject_qualification_request(QualificationRequestId=request_id)

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