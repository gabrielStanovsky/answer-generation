import boto3
import json
import os
from pprint import pprint

config = json.load(open('/home/tony/.aws/mycredentials'))
endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
client = boto3.client('mturk', endpoint_url=endpoint_url, 
					  region_name=config['region_name'],
                      aws_access_key_id=config['aws_access_key_id'],
                      aws_secret_access_key=config['aws_secret_access_key'])

DATASETS = ['cosmosqa', 'drop', 'mcscript', 'narrativeqa', 'quoref', 'ropes', 'socialiqa']
TEST_DURATION = 3600 # 60 minutes to do test

def main():
	# First get a list of all the qualifications that have been created
	response = client.list_qualification_types(MustBeRequestable=True, MustBeOwnedByCaller=True)
	assert response['ResponseMetadata']['HTTPStatusCode'] == 200
	created_qual_names = [q['Name'] for q in response['QualificationTypes']]

	# Create qualifications for datasets if they aren't created and a test is available
	for name in DATASETS:
		# Test if the qual name was already created
		if name in created_qual_names:
			print('Already created', name.upper(), 'qual\n')
			continue

		# Test if the test file exists
		test_file = os.path.join(os.getcwd(), 'mt_html', name + '_qual.xml')
		if not os.path.isfile(test_file):
			print('Test file', test_file, 'not found\n')
			continue

		print('Attemping to create', name.upper(), 'qual')
		response = client.create_qualification_type(Name=name, 
													Test=open(test_file).read(),
													Description='Good at ' + name,
													QualificationTypeStatus='Active', 
													TestDurationInSeconds=TEST_DURATION,
													RetryDelayInSeconds=1)

		if response['ResponseMetadata']['HTTPStatusCode'] != 200:
			print('Could not create qual. Response metadata:')
			pprint(response['ResponseMetadata'])
		else:
			print('Successfully created', name.upper(), 'qual')
		print()

if __name__ == '__main__':
	print()
	main()