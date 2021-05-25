#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import os
import sys
import math
import collections
import codecs
import re


# In[20]:


training_path = "train/"
testing_path = "test/"
folders = ['spam','ham']

stop_words = ["a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"]

bias=0
lrn_rt=0.001
lamda=0.01
itr_count=100


# In[21]:


count_ham=0
count_spam=0
ham_wordsList=[]
spam_wordsList=[]

count_ham_test=0
count_spam_test=0
ham_wordsList_test=[]
spam_wordsList_test=[]


# In[22]:


def get_Count_WordsList(filepath):
    word_Lst = list()
    file_Count = 0
    for files in os.listdir(filepath):
        if files.endswith(".txt"):
            file_Count+=1
            word_Lst += get_Words(files,filepath)
    return file_Count, word_Lst


# In[23]:


def get_Words(file,filepath):
    fileHandler = codecs.open(filepath+"\\" + file,'rU','latin-1')
    Findwords = re.findall('[A-Za-z0-9\']+', fileHandler.read())
    allwords = list()
    for word in Findwords:
        word = word.lower()
        allwords+=[word]
    fileHandler.close()    
    return allwords


# In[24]:


count_spam,spam_wordsList = get_Count_WordsList(training_path+folders[0])
count_ham,ham_wordsList = get_Count_WordsList(training_path+folders[1])

count_spam_test,spam_wordsList_test = get_Count_WordsList(testing_path+folders[0])
count_ham_test,ham_wordsList_test = get_Count_WordsList(testing_path+folders[1])


# In[25]:


def remove_StopWords():
    for wrd in stop_words:
        if wrd in ham_wordsList:
            i = 0
            ham_len=len(ham_wordsList)
            while (i < ham_len):
                if (ham_wordsList[i] == wrd):
                    ham_wordsList.remove(wrd)
                    ham_len = ham_len - 1
                    continue
                i = i + 1
        if wrd in spam_wordsList:
            i = 0
            spam_len=len(spam_wordsList)
            while (i < spam_len):
                if (spam_wordsList[i] == wrd):
                    spam_wordsList.remove(wrd)
                    spam_len = spam_len - 1
                    continue
                i = i + 1
        if wrd in ham_wordsList_test:
            i = 0
            ham_len_test=len(ham_wordsList_test)
            while (i < ham_len_test):
                if (ham_wordsList_test[i] == wrd):
                    ham_wordsList_test.remove(wrd)
                    ham_len_test = ham_len_test - 1
                    continue
                i = i + 1
        if wrd in spam_wordsList_test:
            i = 0
            spam_len_test=len(spam_wordsList_test)
            while (i < spam_len_test):
                if (spam_wordsList_test[i] == wrd):
                    spam_wordsList_test.remove(wrd)
                    spam_len_test = spam_len_test - 1
                    continue
                i = i + 1
                


# In[26]:


#remove_StopWords()


# In[27]:


all_wordsList = ham_wordsList + spam_wordsList
dict_allWords = collections.Counter(all_wordsList)
list_key = list(dict_allWords.keys())
y_list = list()  
file_count = count_spam + count_ham

all_wordsList_test = ham_wordsList_test + spam_wordsList_test
dict_allWords_test = collections.Counter(all_wordsList_test)
list_key_test = list(dict_allWords_test.keys())
y_list_test = list()  
file_count_test = count_spam_test + count_ham_test


# In[28]:


def feature_matrix_init(rw, col):
    x_mat = [0] * rw
    for i in range(rw):
        x_mat[i] = [0] * col
    return x_mat

row_mat = 0
row_mat_test = 0

def create_mat(x_mat, path, list_key, row_mat, cls, y_list):
    for file in os.listdir(path):
        words = get_Words(file, path)
        temp = dict(collections.Counter(words))
        for key in temp:
            if key in list_key:
                column = list_key.index(key)
                x_mat[row_mat][column] = temp[key]
        if (cls == folders[1]):
            y_list[row_mat] = 0
        elif (cls == folders[0]):
            y_list[row_mat] = 1
        row_mat += 1
    return x_mat, row_mat, y_list


# In[29]:


x_mat_train = feature_matrix_init(file_count, len(list_key))
x_mat_test = feature_matrix_init(file_count_test, len(list_key_test))


sgmd_list = list()  
for i in range(file_count):
    sgmd_list.append(-1)
    y_list.append(-1)

for i in range(file_count_test):
    y_list_test.append(-1)

wt_feature = list()

for feature in range(len(list_key)):
    wt_feature.append(0)
    
    
x_mat_train, row_mat, y_list = create_mat(x_mat_train, training_path+folders[1], list_key, row_mat,
                                                       folders[1], y_list)
x_mat_train, row_mat, y_list = create_mat(x_mat_train, training_path+folders[0], list_key, row_mat,
                                                       folders[0], y_list)

x_mat_test, row_mat_test, y_list_test = create_mat(x_mat_test, testing_path+folders[1], list_key_test,
                                                              row_mat_test, folders[1], y_list_test)
x_mat_test, row_mat_test, y_list_test = create_mat(x_mat_test, testing_path+folders[0], list_key_test,
                                                              row_mat_test, folders[0], y_list_test)


# In[30]:


def log_reg_training(file_count, x_count, x_mat_train, y_list):
    calc_sgmdFunc(file_count, x_count, x_mat_train)
    calc_wtUpdt(file_count, x_count, x_mat_train, y_list)


# In[31]:


def calc_wtUpdt(file_count, x_total, x_mat, y_list):
    global sgmd_list
    for x in range(x_total):
        wt = bias
        for fls in range(file_count):
            freq = x_mat[fls][x]
            y = y_list[fls]
            sgmd_val = sgmd_list[fls]
            wt += freq * (y - sgmd_val)

        wt_old = wt_feature[x]
        wt_feature[x] += ((wt * lrn_rt) - (lrn_rt * lamda * wt_old))

    return wt_feature


# In[32]:


def calc_sgmdFunc(file_count, x_total, x_mat):
    global sgmd_list
    for fl in range(file_count):
        sum1 = 1.0
        for x in range(x_total):
            sum1 += x_mat[fl][x] * wt_feature[x]
        sgmd_list[fl] = clac_sgmd(sum1)


# In[33]:


def clac_sgmd(z):
    d = (1 + np.exp(-z))
    sgmd = 1 / d
    return sgmd


# In[34]:


def log_reg_classify():
    true_pred_ham = 0
    false_pred_ham = 0
    true_pred_spam = 0
    false_pred_spam = 0
    j=0
    for file in range(file_count_test):
        sum1 = 1.0
        for i in range(len(list_key_test)):
            word = list_key_test[i]

            if word in list_key:
                index = list_key.index(word)
                wt = wt_feature[index]
                wordcount = x_mat_test[file][i]

                sum1 += wt * wordcount

        sum_sgmd = clac_sgmd(sum1)
        if (y_list_test[file] == 0):
            if sum_sgmd < 0.5:
                true_pred_ham += 1.0
            else:
                false_pred_ham += 1.0
        else:
            if sum_sgmd >= 0.5:
                true_pred_spam += 1.0
            else:
                false_pred_spam += 1.0
        print('Classification of test email '+str(j+1)+'complete.')
        j += 1
    print("Results:\n")
    print("Ham Accuracy of Logistic Regression model:" + str((true_pred_ham / (true_pred_ham + false_pred_ham)) * 100))
    print("Spam Accuracy of Logistic Regression model:" + str((true_pred_spam / (true_pred_spam + false_pred_spam)) * 100))
    print("Overall Accuracy of Logistic Regression model:" + str(((true_pred_ham+true_pred_spam) / (true_pred_ham + false_pred_ham+true_pred_spam + false_pred_spam)) * 100))


# In[ ]:


print("Training of Logistic Regression model started... ")
for i in range(int(itr_count)):
    print(i, end=' ')
    log_reg_training(file_count, len(list_key), x_mat_train, y_list)
print("Training completed successfully")
print("\nClassification of Testing Data started...")
log_reg_classify()


# In[ ]:




