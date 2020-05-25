import nltk
from nltk.util import ngrams

N = 2
#reading from file to string

with open("TEXTFILE.txt" ,"r") as textFile:
    text_body = textFile.read()


from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.lm.preprocessing import pad_both_ends

#create padded sentences
plain_sentences = sent_tokenize(text_body)
word_list_per_sentence = list()
for i in range(len(plain_sentences)):
    word_list_per_sentence.append(list(word_tokenize(plain_sentences[i])))

padded_sentences = list()
for i in range(len(word_list_per_sentence)): 
    padded_sentence = list(pad_both_ends(word_list_per_sentence[i], n = N))
    padded_sentences.append(padded_sentence)


from nltk.lm.preprocessing import flatten
flat_word_list = list(flatten(padded_sentences))

#modelling
from nltk.lm.preprocessing import padded_everygram_pipeline
train,vocab = padded_everygram_pipeline(N, word_list_per_sentence)

from nltk.lm import MLE

lm = MLE(N)
lm.fit(train,vocab)


#matrix
unique_unigrams = list(set(ngrams(flat_word_list,1)))
unique_conditions = list(set(ngrams(flat_word_list,N-1)))


matrix = list()
for i in range(len(unique_conditions)): #rows
    matrix.append(list())
    for unigram in unique_unigrams: #columns
        matrix[i].append(lm.score(unigram[0],unique_conditions[i])) 


        
        

