import sys

def main():
	fp = open(sys.argv[1]).readlines()
	fp1 = open("labels_1.txt",'w')
	for f in fp:
		t_l = f.split("text\":")[1]
		if "\"label\"" in t_l: 
			t = t_l.split("label\":")
			tweet = t[0][1:-3]
			words = tweet.split()
			if len(words) > 7:
				print tweet
				if "-" in t[1]:
					fp1.write(t[1][0:2]+'\n');
				else:
					fp1.write(t[1][0]+'\n');
	fp1.close()
		


if __name__ == '__main__':
	main()
