import merge_cosmosqa
import merge_drop
import merge_mcscript
import merge_narrativeqa
import merge_quoref
import merge_ropes
import merge_socialiqa  

if __name__ == '__main__':
	print('merging cosmosqa...')
	merge_cosmosqa.main()
	print('merging drop...')
	merge_drop.main()
	print('merging mcscript...')
	merge_mcscript.main()
	print('merging narrativeqa...')
	merge_narrativeqa.main()
	print('merging quoref...')
	merge_quoref.main()
	print('merging ropes...')
	merge_ropes.main()
	print('merging socialiqa...')
	merge_socialiqa.main() 