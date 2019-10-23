from glob import glob
from os.path import basename, join

def main():
	writer = open('mt_html/links.md', 'w')
	writer.write('# Links to Mechanical Turk HIT pages\n')

	# Get list of directories of datasets
	dataset_dirs = glob('mt_html/*/')

	# First dataset directories, get links
	for dataset_dir in dataset_dirs:
		dataset = dataset_dir.split('/')[1]
		writer.write('### ' + dataset.upper())

		# Write HTML rendered page to markdown file
		html_preview_extension = 'https://htmlpreview.github.io/?https://github.com/anthonywchen/answer-generation/blob/master/mt_html/'
		for i, hit in enumerate(sorted(glob(join(dataset_dir, '*')))):
			writer.write(' [' + str(i) + ']' + '(' + join(html_preview_extension, dataset, basename(hit)) + ') ')
		writer.write('\n')

	writer.close()

if __name__ == '__main__':
	main()