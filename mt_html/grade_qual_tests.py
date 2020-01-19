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
DATASETS = ['cosmosqa', 'drop', 'mcscript', 'narrativeqa', 'quoref', 'ropes', 'socialiqa']

def get_qual_requests(qual_id):
	response = client.list_qualification_requests(QualificationTypeId=qual_id)
	assert response['ResponseMetadata']['HTTPStatusCode'] == 200
	return response['QualificationRequests']

def grade_qual_requests(qual_id, qual_requests, answer_dict):
	for request in qual_requests:
	    worker_id = request['WorkerId']
	    answer_dict = xmltodict.parse(request['Answer'])['QuestionFormAnswers']['Answer']
	    question_id = answer_dict['QuestionIdentifier']
	    answer_selection = int(answer_dict['SelectionIdentifier'])

def test_cosmosqa(qual_id):
	pass

def test_drop(qual_id):
	pass

def test_mcscript(qual_id):
	pass

def test_narrativeqa(qual_id):
	answer_dict = {'1': [1, 2]}
	qual_requests = get_qual_requests(qual_id)
	grade_qual_requests(qual_id, qual_requests, answer_dict)

def test_quoref(qual_id):
	pass

def test_ropes(qual_id):
	pass

def test_socialiqa(qual_id):
	pass

def main():
	# First get a list of all the qualifications
	response = client.list_qualification_types(MustBeRequestable=True, MustBeOwnedByCaller=True)
	assert response['ResponseMetadata']['HTTPStatusCode'] == 200
	qual_dict = {d['Name']:d['QualificationTypeId'] for d in response['QualificationTypes']}

	# For the qualifications in `DATASETS`, assess the test
	for qual_name, qual_id in qual_dict.items():
		assert qual_name in DATASETS
		if qual_name == 'cosmosqa':
			test_cosmosqa(qual_id)
		elif qual_name == 'drop':
			test_drop(qual_id)
		elif qual_name == 'mcscript':
			test_mcscript(qual_id)
		elif qual_name == 'narrativeqa':
			test_narrativeqa(qual_id)
		elif qual_name == 'quoref':
			test_quoref(qual_id)
		elif qual_name == 'ropes':
			test_ropes(qual_id)
		elif qual_name == 'socialiqa':
			test_socialiqa(qual_id)
		else:
			print('error')

if __name__ == '__main__':
	main()