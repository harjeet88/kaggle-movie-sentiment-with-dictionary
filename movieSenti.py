from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import numpy as np
import pandas as pd

#from senti_classifier import senti_classifier
def run() :
	#sentences = ['The movie was the worst movie', 'It was the worst acting by the actors']
	#pos_score, neg_score = senti_classifier.polarity_scores(sentences)
	#print pos_score, neg_score
	data = pd.read_csv('train.tsv' , delimiter= '\t')
	print data.info()
	
	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(data['Phrase'])
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	clf = MultinomialNB().fit(X_train_tfidf, data['Sentiment'])			
#	clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5)	

	test_data = pd.read_csv('test.tsv' , delimiter= '\t')
	X_new_counts = count_vect.transform(test_data['Phrase'])
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)
	
	predicted = clf.predict(X_new_tfidf)

	output = open('./output-nb.csv', 'w')
	output.write("PhraseId")
	output.write(",")
	output.write("Sentiment")
	output.write("\n")

	for doc, category in zip(test_data['PhraseId'], predicted):
		#print('%r , %s' % (doc, category))
		output.write(str(doc))
		output.write(',')
		output.write(str(category))
		output.write('\n')
	
	output.close()

	print('==========END============')

run()



