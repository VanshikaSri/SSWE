import cv2
import numpy as np

fv = []

import csv 

l = open('trainLabels.txt','r')
klbl = l.readlines()
klbls = []
for i in klbl:
	klbls.append(int(i[:-1]))


with open('train.tsv','rb') as tsvin :
    trainFeat = csv.reader(tsvin, delimiter='\t')
    for i in trainFeat:
    	numb = []
    	for j in i:
		if j != '':
		 	numb.append(float(j))
	fv.append(numb)


tef = []
with open('test.tsv','rb') as tsvin :
    testFeat = csv.reader(tsvin, delimiter='\t')
    for i in testFeat:
    	numb = []
    	for j in i:
		if j != '':
	 		numb.append(float(j))
	tef.append(numb)

fv = np.array(fv)
fv = np.float32(fv)

tef = np.array(tef)
tef = np.float32(tef)
#print fv
klbls = np.array(klbls)
#print klbls 



SZ=20
#bin_n = 16 # Number of bins
 
svm_params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C=2.67, gamma=5.383 )
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
trainData = fv
responses = klbls
svm = cv2.SVM()
svm.train(trainData,responses, params=svm_params)

results = svm.predict_all(tef)
print 'resulting labels'
for i in results :
	print int(i)
