from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from pprint import pprint
import numpy as np
from itertools import islice
import sys

text_train = fetch_20newsgroups(remove=("headers","footers","quotes"), subset='train', shuffle=True, random_state=9)
text_test = fetch_20newsgroups(remove=("headers","footers","quotes"), subset='test', shuffle=True, random_state=42)
target_name = "rec.sport.baseball"
for n, word in enumerate(text_train['target_names']):
    if word == target_name:
        target_index = n

target_data = [d for d, t in zip(text_train['data'],text_train['target']) if t == target_index]
print(len(target_data)) 
target_indexes = [target_index] * len(target_data)
print(len(target_indexes))
#print( text_train['target_names'] )
#print(len(text_train['data']))
#print(len(text_train['target_names']))
#print(len(text_train.target))
#print(text_train['target_names'][text_train.target[0]])
#print(text_train['target_names'][text_train.target[1]])
#sys.exit()
#for key,value in islice(text_train.items(),None):
#    print(f"text_train key is '{key}'")
#    #print(f"text_train data[{key}] is {value}")
#sys.exit()


#pprint(text_train.data[1])
#vectorizer = TfidfVectorizer(max_df=0.5,min_df=5,stop_words="english")
vectorizer = CountVectorizer(max_df=0.7,min_df=1)
vectors = vectorizer.fit_transform(target_data)
word_cloud_vector = vectors.sum(axis=0)
print(word_cloud_vector.shape)
max = (0,0)
for n, freq in enumerate(np.nditer(word_cloud_vector)):
    if freq > max[1]:
        max = (n,freq)

print(vectorizer.get_feature_names_out()[max[0]],max[1])

for word in islice(vectorizer.get_feature_names_out(),5000,5005):
    print(word)
print(f"n_samples: {word_cloud_vector.shape[0]}, n_features: {word_cloud_vector.shape[1]}")
sys.exit()
clf = MultinomialNB().fit(vectors, text_train.target)

#doc_new = ['Where did all the good girls go', 'Did anyone lose a notebook']
x_new_vectors = vectorizer.transform(text_test.data)
predicted = clf.predict(x_new_vectors)
print(np.mean(predicted == text_test.target))

#for doc, category in zip(doc_new, predicted):
#    print(f"{doc} => {text_train.target_names[category]}")