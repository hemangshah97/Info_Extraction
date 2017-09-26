# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:35:07 2017

@author: Admin
"""


#Classification  
#BOW extraction 
def bow_fea(): 
    import nltk 
   # f=open('sample.txt','r') 
    f=open('tier_dataset.txt','r') 
    content=f.read().splitlines() 
    #print content 
    fw=open('feature.txt','w') 
    bow = [] 
    for sentence in content: 
        first=sentence.split(' ', 1)[0] 
        sentence=sentence.split(' ', 1)[1] 
        nouns = []  
        nouns.append(first) 
        words = nltk.word_tokenize(str(sentence)) 
        for word,pos in nltk.pos_tag(words): 
            if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'): 
                nouns.append(word) 
                bow.append(word) 
              
        print nouns 
        fea=' '.join(nouns) 
          
        fw.write(fea) 
        fw.write('\n') 
     
    bow = list(set(bow))  
    print ''          #print featureset 
    print ('Bag of Words are') 
    print bow 
    return bow 
 
    fw.close() 
    f.close() 
 
 
#Naive Bayes Prediction using NLTK 
def pred(): 
    import nltk 
    fr=open('feature.txt','r') 
    content=fr.read().splitlines() 
    #print content 
    feature=[] 
    featuresets=[] 
    fs={} 
    for sentence in content: 
        fs={} 
        label=sentence.split(' ', 1)[0] 
        fea=sentence.split(' ', 1)[1] 
        token = nltk.word_tokenize(fea) 
        #print token 
        for word in token: 
            fs.update({word : 1}) 
        feature=(fs, label) 
        featuresets.append(feature) 
    print '' 
    print 'Feature Vector in Dict Form' 
    print featuresets 
    fr.close() 
    classifier = nltk.NaiveBayesClassifier.train(featuresets) 
    print '' 
    print 'Class Label: ', (classifier.classify({'play' : 1, 'ball' : 1, 'game' : 1})) 
    print 'Class Label: ', (classifier.classify({'hardware' : 1, 'team' : 1, 'program' : 1})) 
     
 
#Feature Extraction for Keras 
def keras_fea(): 
    import nltk 
    bow = bow_fea() 
    fr=open('feature.txt','r') 
    content=fr.read().splitlines() 
    featuresets=[] 
    for sentence in content: 
        fea_vector=[] 
        for i in bow: 
            fea_vector.append(0) 
        label=sentence.split(' ', 1)[0] 
        fea=sentence.split(' ', 1)[1] 
        token = nltk.word_tokenize(fea) 
        for word in token: 
            pos=bow.index(word) 
            fea_vector[pos]=1 
        fea_vector.append(label) 
        print fea_vector 
        featuresets.append(fea_vector) 
    print featuresets 
    fr.close() 
