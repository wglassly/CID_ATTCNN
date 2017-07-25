import os,sys


'''
Usage: svm_format_texts(['a text','b text',...,''], mode='train')
test must follow the train without any other train
'''
def svm_format_texts(text_list,label_list=[], mode = 'train'):
	from libshorttext.libshorttext.converter import *
	sfm_tmp_file = "./tmp/sfm_tmp_file"
	text_converter_dir = sfm_tmp_file + '.text_converter'
	output = "./tmp/" + mode + "_context.svm"
	text_str = '\n'.join(["1" +'\t' +  items  for items in text_list]) if label_list == [] else \
    '\n'.join([str(label) +'\t' + items for label,items in zip(label_list,text_list)])
	open(sfm_tmp_file,'w').write(text_str)
	converter_arguments = '-stopword 0 -stemming 0 -feature 1'
	text_converter = Text2svmConverter(converter_arguments)
	if mode == 'test':
		text_converter.load(text_converter_dir)
	convert_text(sfm_tmp_file, text_converter, output)
	#text_converter.merge_svm_files(output, extra_svm_files)
	if mode == 'train':
		text_converter.save(text_converter_dir)
	return output




'''
load svmfile to csr matrix or dense matrix
'''
def svm_format_load(svm_file, x_format = 'CSR'):
    from sklearn.datasets import load_svmlight_file
    x, y = load_svmlight_file(svm_file)
    if x_format == 'array':
        x = x.todense()
    return x,y


'''
text to array
'''
def text2matrix(text_list,label_list=[], mode = 'train', x_format = 'CSR'):
    output = svm_format_texts(text_list=text_list,label_list=label_list, mode =mode)
    x, y = svm_format_load(output,x_format = x_format)
    return x,y
