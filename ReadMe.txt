# SSWE

Learning sentiment-specific word representations from tweets

Most existing algorithms for learning continuous word representations typically only model the syntactic con-
text of words, while ignoring the sentiment of text. This is problematic for sentiment analysis as words with similar syntactic context but opposite sentiment polarity, such as good and bad, are mapped to neighboring word vectors. We address this issue by learning Sentiment Specific Word Embedding (SSWE), which encodes sentiment information in the continuous representation of words. Specifically, we develop neural networks that effectively incorporate the supervision from sentiment polarity of text (tweets) in their loss functions. Following this, we make use of these word embeddings to train an SVM to predict the sentiment polarity of tweets.

We have used LUA and Python to implement the neural network and train SVM.

Description of Files :

Report.pdf : Report

data/raw_data/train.tsv : Training data provided for training the SVM.				
data/raw_data/test.tsv  : Test data to predict the polarity of tweets.

data/nn_data/tweets.txt : Tokenized tweets (taken from corpus of 10 million tweets) after removing @user, URLs.
data/nn_data/labels.txt : Contains labels of corresponding tweets (taken from corpus of 10 million tweets).
data/nn_data/uni.txt 	: Unigrams of above pre-processed tweet corpus.	
data/nn_data/bi.txt 	: Bigrams of above pre-processed tweet corpus.
data/nn_data/tri.txt    : Trigrams of above pre-processed tweet corpus.

data/features/train.tsv : Feature vectors of tweets in data/test.tsv produced using word embeddings learned using nn.
data/features/test.tsv 	: Feature vectors of tweets in data/train.tsv produced using word embeddings learned using nn.

data/svm_data/trainListTweets.txt : Tokenized tweets (taken from raw_data/train.tsv) after removing @user, URLs.		
data/svm_data/testListTweets.txt  : Tokenized tweets (taken from raw_data/test.tsv) after removing @user, URLs.
data/svm_data/trainLabels.txt     : Contains labels of corresponding svm_data/trainListTweets tweets.

data/output/testLabels.txt  	  : Polarity Sentiment of test tweets predicited by SVM using word embeddings.
data/output/word_embeddings_1.txt : Word Embeddings for vocabulary of tweet corpus provdided learned using nn.
data/output/word_embeddings_2.txt : Word Embeddings for vocabulary of tweet corpus provdided learned using nn.
																						


main/dnn.lua : Lua program used to implement Deep learning neural network on the tweet corpus and produce word embeddings as 				   feature vectors.
main/svm.py  : Python program to train SVM using the word embeddings learned in dnn.lua and predict sentiment polarity of new 				   tweets.


parser/generateVocab.py  : Python program to generate vocabulary from the given tweet corpus.
parser/ngram.py          : Python program to get uni/bi/tri grams of tweet corpus.
parser/parse.py          : extracts tweets and labels from corpus and store in different file.
parser/readTSVTweets.py  : Read .tsv files and store labels and tweets in txt files.
parser/runParsernTokenizer.sh  : Script file 
parser/twokenize.py      : Data cleaning done by tokenizing the tweets and removing @user and URLS from them. 
