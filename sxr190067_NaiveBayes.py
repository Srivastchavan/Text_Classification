#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
import os
import sys
import math
import collections
import codecs
import re


# In[17]:


training_path = "train/"
testing_path = "test/"
folders = ['spam','ham']

stop_words = ["a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"]


# In[18]:


count_ham=0
count_spam=0
ham_wordsList=[]
spam_wordsList=[]


# In[19]:


def get_Count_WordsList(filepath):
    word_Lst = list()
    file_Count = 0
    for files in os.listdir(filepath):
        if files.endswith(".txt"):
            file_Count+=1
            word_Lst += get_Words(files,filepath)
    return file_Count, word_Lst


# In[20]:


def get_Words(file,filepath):
    fileHandler = codecs.open(filepath+"\\" + file,'rU','latin-1')
    Findwords = re.findall('[A-Za-z0-9\']+', fileHandler.read())
    allwords = list()
    for word in Findwords:
        word = word.lower()
        allwords+=[word]
    fileHandler.close()    
    return allwords


# In[21]:


count_spam,spam_wordsList = get_Count_WordsList(training_path+folders[0])
count_ham,ham_wordsList = get_Count_WordsList(training_path+folders[1])

dict_spam = dict(collections.Counter(wrds.lower() for wrds in spam_wordsList))
dict_ham = dict(collections.Counter(wrds.lower() for wrds in ham_wordsList))

all_wordsList = spam_wordsList + ham_wordsList
dict_allWords = collections.Counter(all_wordsList)


# In[22]:


def setCountZero(all_words,spam_ham_dict):
    for wrds in all_words:
        if wrds not in spam_ham_dict:
            spam_ham_dict[wrds] = 0
            
            
setCountZero(dict_allWords,dict_ham)
setCountZero(dict_allWords,dict_spam)


# In[23]:


def remove_StopWords():
    for wrd in stop_words:
        if wrd in dict_ham:
            del dict_ham[wrd]
        if wrd in dict_spam:
            del dict_spam[wrd]
        if wrd in dict_allWords:
            del dict_allWords[wrd] 


# In[24]:


def find_Prob(cls):
    if cls == folders[0]:
        prob_Spam = count_spam/(count_spam + count_ham)
        return prob_Spam
    else:
        prob_Ham = count_ham/(count_spam + count_ham)
        return prob_Ham


# In[25]:


ham_wrds_probability = dict()
spam_wrds_probability = dict()
def calculate_Word_Probability(cls):
    Counter = 0                 
    if cls == folders[1]:
        for wrd in dict_ham:
            Counter += (dict_ham[wrd] + 1)
        for wrd in dict_ham:
            ham_wrds_probability[wrd] = math.log((dict_ham[wrd] + 1)/Counter ,2)
    elif cls == folders[0]:
        for wrd in dict_spam:
            Counter += (dict_spam[wrd] + 1)
        for wrd in dict_spam:
            spam_wrds_probability[wrd] = math.log((dict_spam[wrd] + 1)/Counter ,2) 


# In[26]:



def predict_NB(path, cls):
    ham_prob = 0 
    spam_prob = 0 
    false_Pred = 0
    file_count = 0
                   
    if cls == folders[0]:
        for fileName in os.listdir(path):
            words =get_Words(fileName,path)
            
            ham_prob = math.log(find_Prob(folders[1]),2)
            spam_prob = math.log(find_Prob(folders[0]),2)
            
            for word in words:
                if word in ham_wrds_probability:
                    ham_prob += ham_wrds_probability[word]
                if word in spam_wrds_probability:
                    spam_prob += spam_wrds_probability[word]
            file_count +=1
            if(ham_prob >= spam_prob):
                false_Pred+=1
    if cls == folders[1]:
        for fileName in os.listdir(path):
            words =get_Words(fileName,path)
            
            ham_prob = math.log(find_Prob(folders[1]),2)
            spam_prob = math.log(find_Prob(folders[0]),2)
                        
            for word in words:
                if word in ham_wrds_probability:
                    ham_prob += ham_wrds_probability[word]
                if word in spam_wrds_probability:
                    spam_prob += spam_wrds_probability[word]
            file_count +=1
            if(ham_prob <= spam_prob):
                false_Pred+=1
    return false_Pred,file_count 


# In[27]:


print("Multinomial Naive Bayes model:")  
print("(Results before removing stop words from testing data)")  

calculate_Word_Probability(folders[0])
calculate_Word_Probability(folders[1]) 

false_pred_ham,total_ham = predict_NB(testing_path+folders[1],folders[1])
false_pred_spam,total_spam = predict_NB(testing_path+folders[0],folders[0])
accuracy_ham = round(((total_ham - false_pred_ham )/(total_ham ))*100,2)
accuracy_spam = round(((total_spam -  false_pred_spam )/(total_spam))*100,2)
total_emails = total_ham + total_spam
false_pred_total = false_pred_ham + false_pred_spam
accuracy_total = round(((total_emails  - false_pred_total )/total_emails)*100,2)

print("\nFiles in Ham Folder: ", total_ham)
print("Files in Spam Folder: ", total_spam)
print("Total number of files: ", total_emails)

print("\nHam Folder")
print("Number of emails correctly classified as Ham: ", total_ham - false_pred_ham)
print("Number of emails wrongly classified as Spam: ",false_pred_ham)
print("\nNaive Bayes Accuracy For Ham Folder:" + str(accuracy_ham) + "%")

print("\nSpam Folder")

print("Number of emails correctly classified as Spam: ", total_spam - false_pred_spam)
print("Number of emails wrongly classified as Ham: ",false_pred_spam)
print("\nNaive Bayes Accuracy For Spam Folder: " + str(accuracy_spam) + "%") 

print("\nMultinomial Naive Bayes model Total accuracy for Test data: " + str(accuracy_total) + "%")


# In[28]:


print("Multinomial Naive Bayes model:")  
print("(Results after removing stop words from testing data)")  

remove_StopWords()
calculate_Word_Probability(folders[0])
calculate_Word_Probability(folders[1]) 

false_pred_ham,total_ham = predict_NB(testing_path+folders[1],folders[1])
false_pred_spam,total_spam = predict_NB(testing_path+folders[0],folders[0])
accuracy_ham = round(((total_ham - false_pred_ham )/(total_ham ))*100,2)
accuracy_spam = round(((total_spam -  false_pred_spam )/(total_spam))*100,2)
total_emails = total_ham + total_spam
false_pred_total = false_pred_ham + false_pred_spam
accuracy_total = round(((total_emails  - false_pred_total )/total_emails)*100,2)

print("\nFiles in Ham Folder: ", total_ham)
print("Files in Spam Folder: ", total_spam)
print("Total number of files: ", total_emails)

print("\nHam Folder")
print("Number of emails correctly classified as Ham: ", total_ham - false_pred_ham)
print("Number of emails wrongly classified as Spam: ",false_pred_ham)
print("\nNaive Bayes Accuracy For Ham Folder:" + str(accuracy_ham) + "%")

print("\nSpam Folder")

print("Number of emails correctly classified as Spam: ", total_spam - false_pred_spam)
print("Number of emails wrongly classified as Ham: ",false_pred_spam)
print("\nNaive Bayes Accuracy For Spam Folder: " + str(accuracy_spam) + "%") 

print("\nMultinomial Naive Bayes model Total accuracy for Test data: " + str(accuracy_total) + "%")


# In[ ]:




