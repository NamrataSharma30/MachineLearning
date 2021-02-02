import bs4 as bs
import urllib.request
import numpy as np
from gradient_descent_logistic_regression import utils as ut
import time
from sklearn.tree import DecisionTreeClassifier
from NLP import Adrian_Namrata_naive_bayes_start as nb

lowercase = ''.join(chr(i) for i in range(97, 123)) + ' '
len_of_words = 0


def get_words(st):
    st = st.lower()
    st = st.replace('\r\n', ' ')
    st = ''.join(c for c in st if c in lowercase)
    words = st.split()
    return words


def return_embeddings(words, representation_length, embeddings_dict):
    embeddings = np.empty((len(words), 2), dtype=object, order='C')
    out_of_vocab = np.zeros(representation_length, dtype=float)
    for i in range(len(words)):
        for key in embeddings_dict.keys():
            if words[i] == key:
                if embeddings_dict[key].shape[0] < representation_length:
                    diff = representation_length - (embeddings_dict[key]).shape[0]
                    np.pad(embeddings_dict[key], (diff//2, diff//2), 'constant')
                    embeddings[i] = [words[i], embeddings_dict[key]]
                elif embeddings_dict[key].shape[0] > representation_length:
                    print("First else")
                    embeddings_dict[key] = embeddings_dict[key][:representation_length]
                    embeddings[i] = [words[i], embeddings_dict[key]]
            else:
                embeddings[i] = [words[i], out_of_vocab]
    return embeddings


url_list = ['http://www.gutenberg.org/files/215/215-h/215-h.htm', 'http://www.gutenberg.org/files/345/345-h/345-h.htm',
            'http://www.gutenberg.org/files/1661/1661-h/1661-h.htm']
lowercase = ''.join(chr(i) for i in range(97, 123)) + ' '

embeddings_dict = {}
with open("glove.6B.50d.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector


def split_train_test(X, y, percent_train=0.9):
    ind = np.random.permutation(X.shape[0])
    train = ind[:int(X.shape[0] * percent_train)]
    test = ind[int(X.shape[0] * percent_train):]
    return X[train], X[test], y[train], y[test]

rep_ = 50
for url in url_list[:3]:
    paragraphs = []
    word_lists = []
    sentence_list = []
    data = urllib.request.urlopen(url).read()
    soup = bs.BeautifulSoup(data, 'html')
    count = 0
    for paragraph in soup.find_all('p'):
        par = paragraph.string
        if par:
            par = par.replace('\r\n', ' ')
            sent = par.split('.')
            for s in sent:
                sentence_list.append(s + '.')
                words = get_words(s)
                word_lists.append(words)
    print('This is the first sentence:')
    print(sentence_list[0])
    print('This is the first word list:')
    print(word_lists[0])

    current_url = []
    current_url.append(url)
    final_emb = np.empty((len(word_lists[0]), 2), dtype=object, order='C')
    y = np.zeros(len(word_lists), dtype=int)
    for i in range(len(sentence_list)):
        sentence_emb = return_embeddings(word_lists[i], rep_, embeddings_dict)
        if i == 0:
            final_emb = sentence_emb
            #sentence_embedding = np.hstack((final_emb, np.full(final_emb.shape, sentence_list[i])))
            y[i] = 1
        else:
            final_emb = np.concatenate((final_emb, sentence_emb))
            #sentence_embedding = np.hstack((final_emb, np.full(final_emb.shape, sentence_list[i])))
            if len(current_url) == 1:
                y[i] = 1
            elif len(current_url) == 2:
                y[i] = 2
            else:
                y[i] = 3
        print(i)
sentence_embedding = (np.delete(final_emb, 0, 1)).reshape(-1)
final_embedding = np.zeros((sentence_embedding.shape[0], rep_), dtype=float)
for i in range(sentence_embedding.shape[0]):
    final_embedding[i, 0:rep_] = sentence_embedding[i][0:rep_]
X_train, X_test, y_train, y_test = split_train_test(final_embedding, y)
print("Done")

model = DecisionTreeClassifier(random_state=0, criterion="entropy", splitter="random")

start = time.time()
model.fit(X_train, y_train)
elapsed_time = time.time() - start
print('Elapsed_time training  {0:.6f} secs'.format(elapsed_time))

pred = model.predict(X_train)
print('Accuracy on training set: {0:.6f}'.format(ut.accuracy(y_train, pred)))

start = time.time()
pred = model.predict(X_test)
print(pred)
elapsed_time = time.time() - start
print('Elapsed_time testing  {0:.6f} secs'.format(elapsed_time))
print('Accuracy on test set: {0:.6f}'.format(ut.accuracy(y_test, pred)))
