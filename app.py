import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file

import math
import nltk
nltk.download('punkt')
import pandas as pd
from nltk import sent_tokenize, word_tokenize
#from nltk.corpus import stopwords


app = Flask(__name__) #Initialize the flask App




complex_suffixes = {


#PAST TENSE simple past tense 1st person singular
1 : ["ಳಿದ್ದೆ","ಳಲಿಲ್ಲ","ಳಿದ್ದೆನ","ಳಿದೆನ"], #---> append ಳು

#simple past tense 1st person plural
2 : ["ದಿದೆವು","ದಲಿಲ್ಲ","ದಿದೆವ"],# ---> append ದು

#simple past tense 2nd person
3 : ["ಯಲಿಲ್ಲ"],

#simple past tense 3rd person plural
4 : ["ಯಾಗಿದ್ದರು","ವಾಗಿದ್ದರು","ತಾಗಿದ್ದರು","ದಾಗಿದ್ದರು","ದಿದ್ದರು","ಲಿಲ್ಲ","ದ್ದರಾ"],

#simple past tense 3rd person singular
5 : ["ಯಲಿಲ್ಲ","ಲಿಲ್ಲ","ದನ","ದನಾ"],


#past perfect tense 1st person singular
6 : ["ದಿದ್ದೆ","ಡಿದ್ದೆ","ರಲಿಲ್ಲ","ದ್ದೆನ","ದ್ದೆನಾ"],

#past perfect tennse 1st person plural
7 : ["ದಿದ್ವಿ","ರಲಿಲ್ಲ","ದಿದ್ವಾ"],

#past perfect, 2nd
8 : ["ದಿದ್ದೆ","ಯುತ್ತಿದ್ದೆ","ತ್ತಿದ್ದವರು","ತ್ತಿದ್ದೆ","ತಿದ್ದೆ","ಯುತ್ತದೆ","ತ್ತದೆ","ಯುತ್ತಿರಲಿಲ್ಲ","ತ್ತಿರಲಿಲ್ಲ","ತಿರಲಿಲ್ಲ","ದಿರಲಿಲ್ಲ","ದ್ದಿದ್ದಾ","ಯುತ್ತಿದ್ದಾ","ತ್ತಿದ್ದಾ"],

#past perfect 3rd plural
9 : ["ದಿದ್ದರು"],

#past perfect 3rd singular
10 : ["ದಿದ್ದ","ದಿದ್ದನು","ದಿದ್ದಳು"],

#PAST CONTINUOUS simple tense 1st singular
11 : ["ತ್ತಿದ್ದೆನೆ"],

#past continuous 1st plural
12 : ["ಯುತ್ತಿದ್ದೆವು","ತ್ತಿದ್ದೆವು","ಯುತ್ತಿದ್ದೆವ","ತ್ತಿದ್ದೆವ"],

#past continuous 2nd
13 : ["ತ್ತಿದ್ದೆ","ತಿರಲಿಲ್ಲ","ತ್ತಿದ್ದ","ತ್ತಿದ್ದಾ"],

#past continuous 3rd plural
14 : ["ತ್ತಿದ್ದರು","ತ್ತಿರಲಿಲ್ಲ","ತ್ತಿದ್ದರ","ತ್ತಿದ್ದಾರಾ"],

#past continuous 3rd singular
15 : ["ಯುತ್ತಿದ್ದನ","ಯುತ್ತಿದ್ದನಾ","ಯುತ್ತಿದ್ದಳು","ಯುತ್ತಿದ್ದನು","ಯುತ್ತಿದ್ದಳ","ಯುತ್ತಿದ್ದನ","ಯುತ್ತಿದ್ದಳೆ","ಯುತ್ತಿದ್ದನೆ","ತ್ತಿದ್ದನ","ತ್ತಿದ್ದನಾ","ತ್ತಿದ್ದಳು","ತ್ತಿದ್ದನು","ತ್ತಿದ್ದಳ","ತ್ತಿದ್ದನ","ತ್ತಿದ್ದಳೆ","ತ್ತಿದ್ದನೆ"],

#PAST PERFECT continuous 1st singular
16 : ["ತ್ತಿದ್ದೆ","ತ್ತಿರಲಿಲ್ಲ","ತ್ತಿದ್ದೆನ","ತ್ತಿದ್ದೆನಾ"],

#past perfect continuous 1st plural
17 : ["ಯುತ್ತಿದ್ದೆವೆ","ತ್ತಿದ್ದೆವೆ","ಯುತ್ತಿದ್ದೆವು","ತ್ತಿದ್ದೆವು"],

#past p continous 2nd
18 : ["ತ್ತಿದ್ದೆ","ತ್ತಿದ್ದೆವು","ತ್ತಿರಲಿಲ್ಲ","ತ್ತಿದ್ದಾ"], #----- not needed

#past p continuous 3rd plural
19 : ["ತ್ತಿದ್ದರು","ತ್ತಿದ್ದರು"],# -------- not needed

#past p continuous 3rd singular
20 : ["ತ್ತಿಲ್ಲ","ತ್ತಿದ್ದಳ","ತ್ತಿದ್ದಳು","ತ್ತಿದ್ದನ","ತ್ತಿದ್ದನು","ತ್ತಿದ್ದಾರೆ"],

#PRESENT TENSE
#simple 1st singular
21 : ["ರುತ್ತೆನೆ","ತ್ತೆನೆ","ದಿಲ್ಲ","ಯಲ್ವಾ"],

#simple 1st plural
22 : ["ರುತ್ತೆವೆ","ರುತ್ತೇವೆ","ರುವುದಿಲ್ಲ","ರುತ್ತೇವ","ರುತ್ತೆವ","ತ್ತೆವೆ","ತ್ತೇವೆ","ವುದಿಲ್ಲ","ತ್ತೇವ","ತ್ತೆವ"],

#simple 2nd
23 : ["ತ್ತೀಯ","ವುದಿಲ್ಲ","ತ್ತಿಯ"],

#simple 3rd plural
24 : ["ತ್ತಾರೆ","ತ್ತಾರ"],

#simple 3rd singular
25 : ["ತ್ತಾನೆ","ತ್ತಾಳೆ","ವುದಿಲ್ಲ"],

#Present perfect 1st singular
26 : ["ದ್ದಿನಿ","ದ್ದೆನೆ","ದಿಲ್ಲ","ತ್ತಿದ್ದೆ","ಲ್ಲವ","ದೆನ"],

#present perfect 1st plural
27 : ["ದ್ದೆವೆ","ದ್ದೆವ"],

#present perfect 2nd
28 : ["ಡಿದ್ದೀಯ"],

#present perfect 3rd plural
29 : ["ತ್ತಿದ್ದಾರ","ತ್ತಿದ್ದಾರೆ"],

#present perfect 3rd singular
30 : ["ಯಾಗಿದೆ","ಯಾಗಿಲ್ಲ"],

#present continuous 1st singluar
31 : ["ತ್ತಿದ್ದೆನೆ","ತ್ತೆನೆ","ತ್ತೇನೆ","ತ್ತಿದ್ದೇನೆ","ತ್ತಿಲ್ಲ","ತ್ತಿದ್ದೆನ"],

#present cntinouus 1st plural
32 : ["ತ್ತಿದ್ದೇವೆ","ತ್ತೇವೆ","ತ್ತಿಲ್ಲ","ತ್ತಿದ್ದೇವೆ","ತ್ತಿದ್ದೇವ"],

#present continous 2nd
33 : ["ಯುತ್ತಿದ್ದೀಯ","ಯುತ್ತೀಯ","ಯುತ್ತಿರುವೆ","ಯುತ್ತಿಲ್ಲ","ಯುವುದಿಲ್ಲ","ತ್ತಿದಿಯ"],

#present ocntinuous 3rd plural
34 : ["ತಿದರೆ","ತ್ತಿದ್ದಾರೆ","ತ್ತಿಲ್ಲ","ತ್ತಿದ್ದಾರ","ತಿರುವರ"],

#present continuous 3rd singular
35 : ["ತ್ತಿದ್ದಾನೆ","ತ್ತಿದ್ದಾಳೆ","ತ್ತಾನೆ","ತ್ತಾಳೆ","ತ್ತಿದ್ದಾನ","ತ್ತಿದ್ದಾಳ","ತ್ತಿಲ್ಲ"],

#PRESENT PERFECT continuous tense 1st singular
36 : ["ತ್ತಿದ್ದೀನಿ","ತ್ತಿರುವೆ","ತ್ತಿಲ್ಲ","ತ್ತಿದ್ದೀನಿ","ತ್ತಿಲ್ಲವೆ","ತ್ತಿದ್ದೇನೆ"],

#present perfect continuous tense 1st plural
37 : ["ತ್ತಿದ್ದೇವೆ","ತ್ತಿರುವ","ತ್ತಿರುವೆವು","ತ್ತಿರುವೆವ","ತ್ತಿದ್ದೇವ","ತ್ತಿದೇವ","ತ್ತಿಲ್ಲವ","ತ್ತಿಲ್ಲವಾ"],

#present perfect continuous 2nd
38 : ["ತ್ತಿದೀಯ","ತ್ತಿಲ್ಲ","ತ್ತಿರುವೆಯ","ತ್ತಿದ್ದೆಯ","ತ್ತಿಲ್ಲವ"],

#present perfect continuous 3rd plural
39 : ["ದಲ್ಲಿದೆ","ಯಲ್ಲಿದೆ","ರಲ್ಲಿದೆ"],

#present perfect continuous 3rd singular
40 : ["ತ್ತಿದ್ದಾನೆ","ತ್ತಿದ್ದಾಳೆ","ತ್ತಿದ್ದಾಳ","ತ್ತಿದ್ದಾನೆ"],

41 : ["ಯಾದರೆ","ಗಾದರೆ","ವುದಾದರೆ","ದಾದರೆ"],

42 : ["ಯಾಗಿಯೇ","ಗಾಗಿಯೇ","ದಾಗಿಯೇ","ವಾಗಿಯೇ"],

43 : ["ವಾದರು","ಗಾದರು","ತಾದರು","ದಾದರು","ಯಾದರು","ರಾದರು","ಲಾದರು","ಳಾದರು","ವಾದರೂ","ಗಾದರೂ","ತಾದರೂ","ದಾದರೂ","ಯಾದರೂ","ರಾದರೂ","ಲಾದರರೂ","ಳಾದರೂ"],

44 : ["ತ್ತಿದ್ದರಂತೆ","ದೊಂದಿಗೆ","ಯೊಂದಿಗೆ","ರೊಂದಿಗೆ"],

45 : ["ಗಿದ್ದನು","ಗಿದ್ದಳು","ಗಿದ್ದರು","ಗಿದ್ದರೂ","ತಾದ್ದನು","ತಾದ್ದಳು","ತಾದ್ದರು","ತಾದ್ದರೂ","ದಾದ್ದನು","ದಾದ್ದಳು","ದಾದ್ದರು","ದಾದ್ದರೂ"],

46 : ["ಯೊಂದೆ","ವೊಂದೆ","ರೊಂದೆ","ವೊಂದ","ಯೊಂದ","ರೊಂದ","ವುದೇ"],

47 : ["ಯುವವರ","ರುವವರ","ಸುವವರ"],

48 : ["ದಲ್ಲೇ","ನಲ್ಲೇ","ನಲ್ಲಿ","ವಲ್ಲಿ","ದಲ್ಲಿ","ದಲ್ಲೂ","ಯಲ್ಲಿ","ರಲ್ಲಿ","ಗಳಲ್ಲಿ","ಳಲ್ಲಿ","ಯಲ್ಲಿನ"],

49 : ["ವವರು","ಯವರು","ನವರು","ರವರು","ದವರು","ವವ","ಯವ","ನವ","ರವ","ದವ"],

50 : ["ಗಾಗಿ","ದಾಗಿ","ವಾಗಿ","ರಾಗಿ","ಯಾಗಿ","ತಾಗಿ","ಕ್ಕಾಗಿ","ವಾಗಿದ್ದು","ವಾಗಿದ್ದ","ಗಾಗಿದ್ದು","ಗಾಗಿದ್ದ","ರಾಗಿದ್ದು","ರಾಗಿದ್ದ","ದಾಗಿದ್ದು","ದಾಗಿದ್ದ","ತಾಗಿದ್ದು","ತಾಗಿದ್ದ"],

51 : ["ರನ್ನ","ನನ್ನ","ಯನ್ನ"],

52 : ["ರನ್ನು","ವನ್ನು","ಯನ್ನು","ಗಳನ್ನೇ","ಗಳನ್ನು","ಳನ್ನು","ದನ್ನು"] ,

53 : ["ವಿರುವ","ರುವ","ದ್ದರೆ","ದ್ದಾರೆ"],

54 : ["ತ್ತಾರಂತೆ","ತ್ತಾಳಂತೆ","ತ್ತಾನಂತೆ","ಗಂತೆ","ದ್ದಂತೆ","ದಂತೆ","ನಂತೆ","ರಂತೆ","ಯಂತೆ","ಗಳಂತೆ","ಳಂತೆ","ವಂತೆ"],

55 : ["ಗಳೆಂದು","ಗಂ","ದ್ದಂ","ದಂ","ಯಂ","ರಂ","ವಂ","ಗಿಂದ","ದಿಂದ","ಯಿಂದ","ರಿಂದ","ನಿಂದ"],

56 : ["ನಿಗೆ","ರಿಗೆ","ಯಿಗೆ","ಕೆಗೆ"],

57: ["ದ್ದೇನೆ","ದ್ದಾನೆ","ದ್ದಾಳೆ","ದ್ದಾರೆ","ದಾಗ"],

58 : ["ವಿದೆ" ,"ದಿದೆ","ತಿದೆ","ಗಿದೆ"],

59 : ["ತ್ತಿರು","ವೆಂದು"],

60 : ["ನನ್ನೂ","ಳನ್ನೂ","ರನ್ನೂ"],

61 : ["ಯಾಯಿತು", "ಗಾಯಿತು","ದಾಯಿತು"],

62 : ["ದ್ದನು","ದ್ದಳು","ಯಿದ್ದರು","ದ್ದರು","ದ್ದರೂ","ಗಳೇ","ಗಳು","ಗಳ","ಗಳಿ","ದಳು","ದಳ","ವೆನು","ವನು","ವೆವು","ವಳು","ವಳ","ವುದು","ಲಾಗು","ಗಳಾದ","ಗಳಿಗೆ"],

63 : ["ವುದಕ್ಕೆ","ಕ್ಕೆ","ಗ್ಗಿ","ದ್ದಿ","ಲ್ಲಿ","ನ್ನು","ತ್ತು"],

64 : ["ವಾಯಿತು","ಗಾಯಿತು","ದಾಯಿತು","ತಾಯಿತು","ಲಾಯಿತು","ನಾಯಿತು"],

65 : ["ವಿದ್ದು","ವೆಂದಾಗ"],

66 : ["ವನ್ನೇ","ವೇಕೆ"],

67 : ["ರಾದ","ವಾದ","ಗಾದ","ಯಾದ","ರಾಗುವ"],

68 : ["ವಾದುದು", "ರಾದುದು","ಗಾದುದು","ಯಾದುದು","ದಾದುದು"],

69 : ["ಯಾರು","ದಾರು","ಗಾರು","ರಾರು"],

70 : ["ಗಳಿಸಿ","ಗಳಿಸು","ಗಳಿವೆ","ಗಳಿವ","ಗಳಿವು"],

71 :  ["ಯು","ದ","ವಿಕೆ","ದೇ","ರು","ಳ","ಳೆ","ಲಿದೆ","ದೆ","ರೆ","ಗೆ","ವೆ","ತೆ","ಗೂ"],

72 :  ["ರದ","ಮದ","ನದ"],

73 :  ["ಡಲು","ಲಾಗುತ್ತದೆ","ಸಲು","ಸಿದ್ದಾಳೆ","ಸಿದಾಗ","ಸಲು","ಸಿದರು","ಸಿದನು","ಸಿದಳು","ಸಿದ್ದೇ","ಕಿದೀನಿ"]

}


add_1 = ["ು"]



global flag
flag = 0

def kannada_root(word, inde):
    global flag

    #checking for suffixes which needs to be retained
    for L in complex_suffixes[72]:
        if len(word) > len(L)+1:
            if word.endswith(L):
                inde.append(72)
                return(word[:-(len(L)-1)], inde)

    #checking for suffixes which needs to retained and modified
    for L in complex_suffixes[73]:
        if len(word) > len(L)+1:
            if word.endswith(L):
                flag=1
                word =word[:-(len(L)-1)]
                word = word+add_1[0]
                inde.append(73)
                return(kannada_root(word, inde))

    #checking for suffixes which must be removed
    L=1
    while L<=70:
        for suffix in complex_suffixes[L]:
            if len(word) > len(suffix)+1:
                    if word.endswith(suffix):
                        flag=1
                        inde.append(L)
                        return(kannada_root(word[:-(len(suffix))], inde))
        L = L+1

    #at last checking for remaining suffixes
    if flag == 0:
        for L in complex_suffixes[71]:
            if len(word)-len(L) >len(L)+1:
                if word.endswith(L):
                    inde.append(71)
                    return(word[:-(len(L))], inde)

    return word, inde


file = open('./stopwords.txt', encoding="utf8")
stop_words=file.read()
#print(stop_words)

def _create_frequency_matrix(sentences):
    """
    we create a dictionary for the word frequency table.
    For this, we should only use the words that are not part of the stopWords array.
    Removing stop words and making frequency table
    Stemmer - an algorithm to bring words to its root word.
    :rtype: dict
    """
    frequency_matrix = {}
    stopWords = set(stop_words)
    #ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            flag = 0
            inde = []
            word, inde = kannada_root(word, inde)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix


def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table

def _create_tf_matrix(freq_matrix, doc_table):
    tf_matrix = {}

    count_words_in_document = len(doc_table)

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        for word, count in f_table.items():
            tf_table[word] = doc_table[word] / count_words_in_document

        tf_matrix[sent] = tf_table

    return tf_matrix


def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    alldocs = pd.read_csv('train.csv',usecols = ['headline'])
    wordcount = 1

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():

            for document in alldocs:
                if word in document:
                    wordcount+=1

            idf_table[word] = math.log10(total_documents / float(wordcount))

        idf_matrix[sent] = idf_table

    return idf_matrix

def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix

def _create_gss_matrix(freq_matrix,category,total_documents):

    #print(category)
    gss_matrix = {}

    alldocs = pd.read_csv('train.csv', usecols = ['headline','label'])
    doccount = total_documents
    #print(alldocs.head())

    for sent,f_table in freq_matrix.items():
        gss_table = {}
        a=0
        b=0
        c=0
        d=0
        for word in f_table.keys():
            for i in range(len(alldocs['headline'])):
                if word in alldocs['headline'][i]:
                    if alldocs['label'][i]==category:
                        a+=1
                    else:
                        b+=1
                else:
                    if alldocs['label'][i]==category:
                        c+=1
                    else:
                        d+=1

            #print(a," ",b," ",c," ",d)
            p1 = float(a/doccount)
            p2 = float(d/doccount)
            p3 = float(c/doccount)
            p4 = float(b/doccount)
            gss_table[word] = float(p1*p2) - float(p3*p4)
            #print(word," ",gss_table[word])

            gss_matrix[sent] = gss_table
    return gss_matrix

def _create_tf_gss_matrix(tf_matrix,gss_matrix):
    tf_gss_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), gss_matrix.items()):

        tf_gss_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_gss_table[word1] = float(value1 * value2)

        tf_gss_matrix[sent1] = tf_gss_table

    return tf_gss_matrix

def _score_sentences(tf_idf_matrix,tf_gss_matrix) -> dict:
    """
    score a sentence by its word's TF-IDF, TF_GSS, Position Of sentence
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """
    alpha  = 0.75
    beta = 0.85
    gamma = 0.15

    sentenceValue1 = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue1[sent] = total_score_per_sentence / count_words_in_sentence

    sentenceValue2 = {}

    for sent, f_table in tf_gss_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue2[sent] = total_score_per_sentence / count_words_in_sentence

    sentenceValue3 = {}
    doc_length = len(sentenceValue1)

    pos = 0
    for sent in sentenceValue2:
        sentenceValue3[sent] = abs(math.cos(pos*2*math.pi/doc_length))
        pos+=1

    for sent in sentenceValue2:
        if sent in sentenceValue1:
            sentenceValue1[sent] = (alpha * sentenceValue1[sent]) + (beta* sentenceValue2[sent]) + (gamma * sentenceValue3[sent])
    else:
        pass

    return sentenceValue1

def _find_average_score(sentenceValue, num_sent) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    '''sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))
    '''

    sort_sent_value = dict(sorted(sentenceValue.items(), key=lambda x: x[1], reverse=True))

    if(num_sent == 0):
        return (list(sort_sent_value.items())[0][1]+1)


    count=0

    for i in sort_sent_value.keys():
        if count==num_sent-1:
            key =  i
            break
        else:
            count+=1

    threshold = sort_sent_value[key]

    return threshold

def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    if sentence_count == 0:
        return "NIL"
    return summary

def run_summarization(text,category,num_sent):
    """
    :param text: Plain summary_text of long article
    :return: summarized summary_text
    """

    '''
    We already have a sentence tokenizer, so we just need
    to run the sent_tokenize() method to create the array of sentences.
    '''
    # 1 Sentence Tokenize
    sentences = sent_tokenize(text)
    #sentences = clean_text(raw_sentences)
    total_length = len(sentences)
    total_documents = 15
    #print(sentences)

    # 2 Create the Frequency matrix of the words in each sentence.
    freq_matrix = _create_frequency_matrix(sentences)
    #print(freq_matrix)

    # 4 creating table for documents per words
    count_doc_per_words = _create_documents_per_words(freq_matrix)
    #print(count_doc_per_words)

    '''
    Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
    '''
    # 3 Calculate TermFrequency and generate a matrix
    tf_matrix = _create_tf_matrix(freq_matrix,count_doc_per_words)
    #print(tf_matrix)


    '''
    Inverse document frequency (IDF) is how unique or rare a word is.
    '''
    # 5 Calculate IDF and generate a matrix
    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    #print(idf_matrix)

    # 6 Calculate TF-IDF and generate a matrix
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
    #print(tf_idf_matrix)

    # 5 Calculate GSS and generate a matrix
    gss_matrix = _create_gss_matrix(freq_matrix, category, total_documents)
    #print(gss_matrix)

    # 6 Calculate TF-GSS and generate a matrix
    tf_gss_matrix = _create_tf_gss_matrix(tf_matrix, gss_matrix)
    #print(tf_gss_matrix)

    # 7 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(tf_idf_matrix,tf_gss_matrix)
    #print(sentence_scores)

    # 8 Find the threshold
    threshold = _find_average_score(sentence_scores,num_sent)
    #print(threshold)

    # 9 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, threshold)
    return summary

txt_str = "Hello"


@app.route('/')

@app.route('/index')
def index():
	return render_template('index.html')


@app.route('/login')
def login():
	return render_template('login.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        #df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)



@app.route('/summarization')
def summarization():
	return render_template("summarization.html")

@app.route('/textsummary', methods=['POST'])
def text_summary():
    if request.method == 'POST':
        # Get the category and Kannada text from the form
        category = request.form.get('category')
        kannada_text = request.form.get('kannadaText')
        text_str = kannada_text
        num_sent = request.form.get('numLines')
        num_sent = int(num_sent)

        result = run_summarization(text_str,category,num_sent)

        # Perform text summarization here
        # You can use the previously discussed summarization code here to generate the summary

        # For demonstration, let's print the category and summarized text
        print(f"Category: {category}")
        print(f"Kannada Text: {kannada_text}")

        # Replace the following line with code to generate the summary
        summary = result
        print(f"summary")

        return render_template('summary.html', category=category, kannada_text=kannada_text, summary=summary)


if __name__ == "__main__":
	#text = open('ai.txt', 'r').read()
    # category = input("Enter Category the document belongs to. Entertainment ? Sports ? Mythology ?").lower()
    # num_sent = int(input("Enter number of sentences the summary should span to ?"))
    # result = run_summarization(text_str,category,num_sent)
    # print()
    # print(result)

    app.run(debug=True)
