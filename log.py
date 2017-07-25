import os,sys
import numpy as np

#context_file: filename of testfile
#predict     : np.array, [1,0, ... , 0] 
#golden      : np.array
def log_error(context_file, predict, golden, log_name='log.txt'):
	miss, wrong = [],[]
	i = 0
	for p,g in zip(predict,golden):
		p1 = 1 if p >= 0.5 else 0
		if p1!=g:
			if g == 1:
				miss.append(i)
			else:
				wrong.append(i)
		i += 1
	context = open(context_file,'r').read().strip().split('\n')
	miss_context, wrong_context = [context[m] for m in miss], [context[w] for w in wrong]
	log = "\n".join(wrong_context) + "*"*20 + "\n" + "\n".join(miss_context)
	open(log_name,'w').write(log)

