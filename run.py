import os,sys


if __name__ == '__main__':
	
	run_name, times, outputfile = sys.argv[1:]

	if run_name.find('pos') != -1:
		for t in range(0,int(times)):
			print "python " + run_name + " ./data/train_full ./data/test_full >> " + outputfile
			os.system("python " + run_name + " ./data/train_full ./data/test_full >> " + outputfile)
	else:
		for t in range(0,int(times)):
			print "python " + run_name + " ./data/train_only_between ./data/test_only_between >> " + outputfile
			os.system("python " + run_name + " ./data/train_only_between ./data/test_only_between >> " + outputfile)