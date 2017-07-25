import os,sys

SGolden = 1066

def f_score(p,r):
	return 2*p*r/(p+r)


def benchmark_libshorttext(result_file):
	result = open(result_file,'r').read().strip().split('\n')[6:]
	print len(result)
	right = 0
	miss = 0
	wrong = 0
	for line in result:
		predict_label, golden_label, pp, gp =  line.split('\t')
		if predict_label == "0" and golden_label == "0":
			continue
		elif predict_label == "1" and golden_label =="1":
			right += 1
		elif predict_label =="1" and golden_label == "0":
			wrong += 1
		elif predict_label =="0" and golden_label =="1":
			miss += 1
	p = float(right)/ (right + wrong)
	r = float(right)/ SGolden
	f = f_score(p, r)
	print p,r,f
	print right +miss

def benchmark_cnn(result_file,golden_file):
	result = open(result_file,'r').read().strip().split('\n')
	golden = open(golden_file,'r').read().strip().split('\n')
	#print len(result)
	right = 0
	miss = 0
	wrong = 0
	for line, golden in zip(result,golden):
		#predict_label =  "1"
		a, b = line.strip().split(' ')
		predict_label = "0" if float(a) >= float(b) else "1"
		golden_label = golden.split('\t')[0]
		if predict_label == "0" and golden_label == "0":
			continue
		elif predict_label == "1" and golden_label =="1":
			right += 1
		elif predict_label =="1" and golden_label == "0":
			wrong += 1
		elif predict_label =="0" and golden_label =="1":
			miss += 1
	p = float(right)/ (right + wrong)
	r = float(right)/ SGolden
	f = f_score(p, r)
	print p,r,f
	print right +miss

def benchmark_cnn(result_file,golden_file):
	result = open(result_file,'r').read().strip().split('\n')
	goldens = open(golden_file,'r').read().strip().split('\n')

	right = 0
	miss = 0
	wrong = 0

	right_list, miss_list, wrong_list = [], [], []
	i = -1
	for line, golden in zip(result,goldens):
		#predict_label =  "1"
		i += 1
		predict_label = "0" if float(line.strip()) <= 0.5 else "1"
		#a, b = line.strip().split(' ')
		#predict_label = "0" if float(a) >= float(b) else "1"
		golden_label = golden.split('\t')[0]
		if golden.find('.') == -1:
			continue
		if predict_label == "0" and golden_label == "0":
			continue
		elif predict_label == "1" and golden_label =="1":
			right += 1
			right_list.append(goldens[i])
		elif predict_label =="1" and golden_label == "0":
			wrong += 1
			wrong_list.append(goldens[i])
		elif predict_label =="0" and golden_label =="1":
			miss += 1
			miss_list.append(goldens[i])

	p = float(right)/ (right + wrong) if right+wrong >0 else 0.01
	r = float(right)/ SGolden
	f = f_score(p, r)
	print p,r,f

	mstr = "\n".join(wrong_list) + "===================================\n" + '\n'.join(miss_list)
	open("log.txt",'w').write(mstr)


def benchmark_cnn_threshold(result_file,golden_file,thresholds = [0.1, 0.2, 0.4, 0.8, 1, 1.2, 1.5, 2, 4, 6, 8 ,10]):
	result = open(result_file,'r').read().strip().split('\n')
	goldens = open(golden_file,'r').read().strip().split('\n')
	for threshold in thresholds:
		right = 0
		miss = 0
		wrong = 0

		right_list, miss_list, wrong_list = [], [], []
		i = -1
		for line, golden in zip(result,goldens):
			#predict_label =  "1"
			i += 1
			#predict_label = "0" if float(line.strip()) <= 0.5 else "1"
			a, b = line.strip().split(' ')
			predict_label = "0" if float(a) >= (float(b)*threshold) else "1"
			golden_label = golden.split('\t')[0]
			if predict_label == "0" and golden_label == "0":
				continue
			elif predict_label == "1" and golden_label =="1":
				right += 1
				right_list.append(goldens[i])
			elif predict_label =="1" and golden_label == "0":
				wrong += 1
				wrong_list.append(goldens[i])
			elif predict_label =="0" and golden_label =="1":
				miss += 1
				miss_list.append(goldens[i])
		p = float(right)/ (right + wrong) if right+wrong >0 else 0.01
		r = float(right)/ SGolden
		f = f_score(p, r)
		print p,r,f, threshold


def benchmark_keras_cnn(result_file, threshold = 0.5):
	result = open(result_file).read().strip().split('\n')
	right, miss, wrong = 0,0,0
	for line in result:
		predict_label , golden_label = line.split('\t')
		predict_label = float(predict_label[1:-1])
		golden_label = int(golden_label)
		predict_label = 1 if predict_label > threshold else 0
		if predict_label == 0 and golden_label == 0:
			continue
		elif predict_label == 1 and golden_label == 1:
			right += 1
		elif predict_label == 1 and golden_label == 0:
			wrong += 1
		elif predict_label == 0 and golden_label == 1:
			miss += 1
	p = float(right)/ (right + wrong)
	r = float(right)/ SGolden
	f = f_score(p, r)
	print p,r,f
	print right +miss



if __name__ == '__main__':
	benchmark_cnn(sys.argv[1], sys.argv[2])
	#benchmark_keras_cnn(result_file=sys.argv[1], threshold=float(sys.argv[2]))
