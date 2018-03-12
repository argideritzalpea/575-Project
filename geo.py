#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 00:14:48 2018

@author: Chris
"""
import nltk
import sys
import os
import re
import math
import inflect
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from word2number import w2n
from nltk.tag import pos_tag
import xml.etree.ElementTree as ET
import glob

pluralizer = inflect.engine()

my_path = sys.argv[1]
files = glob.glob(my_path + '/**/*.xml', recursive=True)

hypos = lambda s:s.hyponyms()
hyper = lambda s:s.hypernyms()

running_numtrue = 0
running_numfound = 0
running_matches = 0

def wrdNum(word):
    try:
        w2n.word_to_num(word)
        return True
    except:
        return False

#print(distlems)

def distDig(synset, targets, untargets):
    distsyns = set()
    distlems = set()
    distinflect = set()
    mhypos = list(synset.closure(hypos))
    for hypo in mhypos:
        defin = set(hypo.definition().split())
        if not defin.isdisjoint(targets):
            if defin.isdisjoint(untargets):
                distsyns.add(hypo)
    for syn in distsyns:
        for lem in syn.lemma_names():
            if wrdNum(lem) == False:
                distlems.add(lem)
    distlems.remove('in')
    for lem in distlems:
        distinflect.add(pluralizer.plural(lem))
    
    return distsyns, distlems, distinflect

for file in files:
    print("")
    print(file)
    input_file = open(file, "r").read().split("\n")
    tree = ET.parse(file)
    root = tree.getroot()
    
    originalspans = []
    for element in root.iter('MEASURE'):
        originalspans.append(element.get('text'))
    
    startind = 0
    endind = 0
    for index, line in enumerate(input_file):
        if re.match('\<TEXT\>\<\!\[CDATA\[', line):
            startind = index
        if re.match('\]\]\>\<\/TEXT\>', line):
            endind = index        
            
    tokenized_docs=[doc for doc in input_file if doc != ""][startind:endind]
    tokenized_docs = " ".join(tokenized_docs)
    tokenized_docs = word_tokenize(tokenized_docs)
    tokenized_docs = [w.lower().strip('.').strip(',') for w in tokenized_docs if re.search('\w', w)]
    #for k in tokenized_docs:
    #    print(k)
    tagged_doc = pos_tag(tokenized_docs)
        
    measure = wn.synset('measure.n.02')
    targets = {'distance', 'space', 'quantity', 'length', 'width'}
    untargets = {'volume', 'contained', 'contain', 'time', 'hold', 'period'}
    distsyns, distlems, distinflect = distDig(measure, targets, untargets)
    
    comparatives = {'less than', 'fewer than', 'greater than', 'more than', 'less then', 'fewer then', 'greater then', 'more then'}
    adjs = {'near', 'close', 'far', 'nearby'}
    degreeadjs = {'many', 'several', 'exactly', 'about', 'approximately', 'nearly', 'almost', 'few', 'over', 'some', 'within'}
    verbtags = {'VBZ', 'VB', 'VBG', 'VBN', 'VBD', 'VBP'}
    
    ### Add approx distances
    spans = []
    for index, word in enumerate(tagged_doc):
        
        if word[0] in adjs:
            if tagged_doc[index-1][1] in verbtags:
                spans.append(word[0])
            elif word[0] in {"near", "nearby", "close"}:
                spans.append(word[0])
        floatnum = 0
        if index >= 1:
            prevT = tagged_doc[index-1][1]
            prevW = tagged_doc[index-1][0]
        try:
            floatnum = float(prevW.replace(',', ''))
        except:
            pass
        if index >= 2:
            prev2T = tagged_doc[index-2][1]
            prev2W = tagged_doc[index-2][0]
            if index >= 3:
                prev3T = tagged_doc[index-3][1]
                prev3W = tagged_doc[index-3][0]
        if index <= len(tagged_doc)-2:
            nextT = tagged_doc[index-1][1]
            nextW = tagged_doc[index+1][0]
        tag = word[1]
        word = word[0]
        if word in distlems.union(distinflect):
            if prevW.isnumeric() or floatnum != 0 or wrdNum(prevW):
                if prev2W in degreeadjs:
                    spans.append((prev2W, prevW, word))
                else:
                    spans.append((prevW, word))
            if prevW in degreeadjs:
                spans.append((prevW, word))
            if index >= 3:
                if prevT == "DT":
                    combinedcomp = prev3W + " " + prev2W
                    #print(combinedcomp)
                    if combinedcomp in comparatives:
                        spans.append((prev3W, prev2W, prevW, word))
            #elif wrdNum(prevW):
            #    spans.append((tagged_doc[index-1][0], word))
        match = re.match(r"([0-9]+)([a-z]+)", word, re.I)
        if match:
            items = match.groups()
            if items[1] in distlems:
                spans.append(word)
    stringspans = []
    #print(spans)
    for i in spans:
        if isinstance(i, str):
            stringspans.append(i)
        else:
            stringspans.append(" ".join(i))
    
    numfound = len(stringspans)
    numtrue = len(originalspans)
    matches = 0
    unmatched = []
    originalspans = [x.lower().strip(" ") for x in originalspans]
    for x in stringspans:
        if x in originalspans:
            theindex = originalspans.index(x)
            originalspans.pop(theindex)
            matches += 1
        else:
            unmatched.append(x)
    numnomatch = len(unmatched)
    print("Unmatched gold standard spans:", originalspans)
    print("Predicted spans:", stringspans)
    
    running_numtrue += numtrue
    running_numfound += numfound
    running_matches += matches
    
    if numtrue == 0:
        recall = 0
    else:
        recall = matches/numtrue
    
    if numfound == 0:
        precision = 0
    else:
        precision = matches/numfound
    
    print('recall', recall)
    print('precision', precision)
    if (precision+recall) == 0:
        print('F-score', 0)
    else:
        print('F-score', (2*precision*recall)/(precision+recall))

finalrecall = running_matches/running_numtrue
finallprec = running_matches/running_numfound
finalF = (2*finallprec*finalrecall)/(finallprec+finalrecall)
print('finalrecall', finalrecall)
print('finallprec', finallprec)
print('finalF', finalF)