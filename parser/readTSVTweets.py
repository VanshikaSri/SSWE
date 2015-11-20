from itertools import repeat
import csv

#twFile = open("trainTweets.txt",'w')
twFile = open("trainLabels2.txt",'w')
with open('/home/vanshika/Downloads/Project3/train.tsv','rb') as tsvin :
    tsvin = csv.reader(tsvin, delimiter='\t')
    for i in tsvin:
 	if i[2] == "-1" or i[2] == "1" or i[2] == "0":
		twFile.write(i[2]+'\n')

twFile.close()


