import sys
import re

def main():
	fp = open(sys.argv[1]).read()
	result = re.sub(r"http\S+", "", fp)
	result1 = re.sub(r"@\S+", "", result)
	result1 = re.sub(r"@", "", result1)
	fp = result1.split()
	a = list(set(fp))
	for i in a:
		print i 

		


if __name__ == '__main__':
	main()
