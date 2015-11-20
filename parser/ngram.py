import nltk
from nltk import bigrams
from nltk import trigrams
import sys


tweets = open(sys.argv[1],'r').readlines()
#labels = open(sys.argv[2],'r').readlines()

uniFile = open("trainUni.txt",'w')
#biFile = open("bi1.txt",'w')
#triFile = open("tri1.txt",'w')


# split the texts into tokens
for i in range(len(tweets)):
	tweet = tweets[i]
#	label = labels[i]	
	tokens = tweet.split()
	tokens = [token.lower() for token in tokens if len(token) > 0] #same as unigrams
#	bi_tokens = bigrams(tokens)
#	tri_tokens = trigrams(tokens)
	
	for uni in tokens:
		uniFile.write(uni+'\n')
	
#	for bi1 in bi_tokens:
#		for bi in bi1:
#			biFile.write(bi+' ')
#		biFile.write(label)
	
#	for tri1 in tri_tokens:
#		for tri in tri1:
#			triFile.write(tri+' ')
#		triFile.write(label)

	

