___
**System Requrement:**

python 2.7, theano, keras, gensim

sklearn	&emsp;&emsp;&emsp; #for load KB features*

**maybe acquired:** *h5py, libshortexts*
___
**How to use**

1. We have extracted all (cheimcal,disease) pairs from BiocreativeV CDR data and stored them in "data/" such as "train_full" . Knowledge-Bases features ("*.svm") also store in that folder.
	    
	&emsp;&emsp;"train_full" and "test_full": full sentences

	&emsp;&emsp;"train_only_between" and "test_only_between": segment between two entities.
 
2. Run:

	Test our model with KB features: 

>       python pos_fea_cnn.py data/train_full data/test_full


	Test our model without KB features: 

> 	python pos_cnn.py data/train_full data/test_full


	Trainditional CNN model with KB features: 

> 	python standard_fea_cnn.py data/train_only_between data/test_only_between


	Trainditional CNN model without KB features: 

>       python standard_cnn.py data/train_only_between data/test_only_between


 Any questions pls connect lhd911107@gmail.com

