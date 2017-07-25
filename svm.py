import os,sys
from benchmark import benchmark_libshorttext as benchmark


def pipeline(mode = "between", feature = "add"):
	if mode == "between":
		trainfile = "./data/train_only_between"
		testfile = "./data/test_only_between"
	else:
		trainfile = "./data/train_full"
		testfile = "./data/test_full"
	if feature == "add":
		model_file = "../data/model/libshorttext.model"
		train_svms = ['./data/train_medi_fea.svm', './data/train_ctd_fea.svm', \
    './data/train_mesh_fea.svm', './data/train_sider_fea.svm',
    './data/train_mention_fea.svm']
		test_svms = ['./data/test_medi_fea.svm', './data/test_ctd_fea.svm', \
    './data/test_mesh_fea.svm', './data/test_sider_fea.svm',
    './data/test_mention_fea.svm']
	else:
		model_file = "../data/model/no_fea_shorttext.model"
		train_svms, test_svms = [], []
	run(trainfile, testfile, model_file,train_svms, test_svms)

def run(trainfile, testfile, model_file,train_svms, test_svms):
	train(trainfile, model_file,train_svms)
	result_file = test(testfile,model_file,test_svms)
	benchmark(result_file)

def train(trainfile, model_file,train_svms):
	feature_file_list =  train_svms
	start_str = "python ./libshorttext/text-train.py -f " 
	feature_str  = "-A " + " -A ".join(feature_file_list) if feature_file_list else ""
	main_str = ' ' + trainfile + " " + model_file
	sys_str = start_str + feature_str + main_str
	print sys_str
	os.system(sys_str)

def test(testfile,model_file, test_svms):
	feature_file_list =  test_svms
	outfile = testfile+'_libshort_result'
	start_str = "python ./libshorttext/text-predict.py -f " 
	feature_str  = "-A " + " -A ".join(feature_file_list) + ' '  if feature_file_list else " "
	main_str = testfile +  " " + model_file + " " + outfile
	sys_str =  start_str + feature_str + main_str
	print sys_str
	os.system(sys_str)
	return outfile

if __name__ == '__main__':
	mode,feature = sys.argv[1:]
	pipeline(mode,feature)
