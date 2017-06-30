#!/usr/bin/python
import numpy as np
from scipy.optimize import linprog
from scipy.stats import poisson
import random
import copy
import pylab as P
import math
import pickle

"""
    Compute coefficient of variance.
"""
def computeGamma(histogram):
    n = np.sum(histogram)
    c = np.sum(histogram > 0)
    f1 = float(np.sum(histogram == 1))
    c_hat = 1. - f1/n
    
    s = 0.
    for i in range(2,len(histogram)):
        s += np.sum(histogram == i) * i * (i-1)
    gamma = s * (c/c_hat) / n / (n-1) - 1.
    
    return max(gamma,0)
    
"""
    Compute sample coverage
"""
def sampleCoverage(data):
    hist  = np.sum(data == 1,axis=1)
    n = float(np.sum(hist))
    f1 = float(np.sum(hist == 1))
    if n == 0:
        return 0.
    
    return 1 - f1/n


"""
    Number of workers needed to get at least 1 vote for all items
"""
def minTasks(data):
    for w in range(1,len(data[0])+1):
        if np.sum(np.sum(data[:,0:w]!=-1,axis=1) > 0) == len(data):
            return w
    return -1

"""
    How many tasks would have been needed, 
    if we have thrown in a fixed number of workers for all items?
"""
def minTasksToCleanAll(data,asgnPerTask=10,workers=3):
    n = len(data) # the total number of items
    return int(math.ceil(float(n)*3/asgnPerTask))

"""
    Number of positive votes
"""
def positiveVotes(data):
    return np.sum(np.sum(data == 1))

"""
    Species estimation, Chao estimator, for the number of errors.
"""
def chao92(data,corrected=True):
    data_subset = copy.deepcopy(data)
    n_worker = []
    for i in range(len(data_subset)):
        n_worker.append(np.sum(data_subset[i] != -1))
    n_worker = np.array(n_worker)
    data_subset[data_subset == -1] = 0
    histogram = np.sum(data_subset,axis=1)
    n = float(np.sum(histogram))
    n_bar = float(np.mean([i for i in histogram if i > 0]))
    v_bar = float(np.var(histogram[histogram>0]))
    d = float(np.sum(histogram > 0)) 
    f1 = float(np.sum(histogram == 1)) 
    if n == 0:
        return d
    c_hat = 1 - f1/n
    gamma = computeGamma(histogram)
    if c_hat == 0.:
        return d
    if corrected:
        est = d/c_hat + n*(1-c_hat)/c_hat*gamma
    else:
        est = d/c_hat
    return est

"""
    Shift-based Chao estimation technique. 
    
"""
def sChao92(data,corrected=True,shift=0):
    data_subset = copy.deepcopy(data)
    n_worker = []
    for i in range(len(data_subset)):
        n_worker.append(np.sum(data_subset[i] != -1))
    n_worker = np.array(n_worker)
    data_subset[data_subset == -1] = 0
    histogram = np.sum(data_subset==1,axis=1)
    n = float(np.sum(histogram))
    n_bar = float(np.mean([i for i in histogram if i > 0]))
    v_bar = float(np.var(histogram[histogram>0]))
    d = np.sum(histogram >(n_worker/2))
    
    f1 = float(np.sum(histogram == 1+shift)) 
    if n == 0:
        return d
    c_hat = 1 - f1/n
    gamma = computeGamma(histogram)
    if c_hat == 0.:
        return d
    if corrected:
        est = d/c_hat + n*(1-c_hat)/c_hat*gamma
    else:
        est = d/c_hat

    return est


"""
    extrapolate from the data (worker-responses or tasks); 
    or from a fraction (0.0<= q <=1.0) of data.
"""
def extrapolateFromSample(data,labels,q=1.0,golden=True): 
    n = q * len(data) # we sample from the entire records, not those reviewed by workers.
    sampleIdx = np.random.choice(range(len(data)),n,replace=False)
    sample = data[sampleIdx,:]
    est = vNominal(sample)/q
    if golden:
        est = np.sum(np.array(labels)[sampleIdx])/q
    return est


"""
    extrapolation from a random n/len(data) sample - taken on the fly from the data
"""
def extrapolation(data,pair_solution,n):
    ilist = pair_solution.keys()
    ext_array = []
    pair_sample = np.random.choice(range(len(ilist)),n,replace=False)
    ext = np.sum([pair_solution[ilist[s]] for s in pair_sample])
    q = len(pair_sample) / float(len(data))
    ext_array.append(ext/q)
    ext = np.mean(ext_array) 
    return ext

"""
    extrapolation based on a golden sample data, for total error estimates.
"""
def extrapolation2(data,d,samples):
    ext_array = []
    for s in samples:
        ext = vNominal(s[:,0:len(data[0])])
        q = np.sum(np.sum(s == -1,axis=1) != -len(s[0])) / float(len(data))
        ext_array.append(ext/q)
    ext = np.mean(ext_array)
    return ext

"""
    extrapolation based on a golden sample data, for switch estimates.
"""
def extrapolation3(data,d,samples):
    ext_array = []
    for s in samples:
        ext = sNominal(s[:,0:len(data[0])])
        q = np.sum(np.sum(s == -1,axis=1) != -1) / float(np.sum(np.sum(d != -1)))
        ext_array.append(ext/q)
    ext = np.mean(ext_array)
    return ext



"""
    observed unique data items. if this converges to GT, then there exists no FP.
"""
def nominal(data):
    return np.sum(np.sum(data == 1,axis=1) > 0)

def nominalCov(data):
    return np.sum(np.sum(data != -1,axis=1) >0)

def nominalF1(data):
    return np.sum(np.sum(data==1,axis=1)==1)

def nominalF2(data):
    return np.sum(np.sum(data==1,axis=1)==2)

"""
    voting based nominal estimation.
"""
def vNominal(data):
    return np.sum(np.sum(data == 1,axis=1) > np.sum(data != -1,axis=1)/2)

def majority_fp(data, slist):
    return np.sum(np.logical_and(np.sum(data == 1,axis=1) > np.sum(data != -1,axis=1)/2,slist == 0))

def majority_fn(data, slist):
    print 'false negative:', np.sum(np.logical_and(np.sum(data == 0,axis=1) >= np.sum(data != -1,axis=1)/2,slist == 1))
    
    return np.sum(np.logical_and(np.sum(data == 0,axis=1) >= np.sum(data != -1,axis=1)/2,slist == 1))


"""
    Observed number of switches.
"""
def sNominal(data,pos_switch=True,neg_switch=True):
    data_subset = data # no copying
    majority = np.zeros((len(data_subset),len(data_subset[0])))
    switches = np.zeros((len(data_subset),len(data_subset[0])))
    for i in range(len(data_subset)):
        prev = 0
        for w in range(0,len(data_subset[0])):
            # the first worker is compared with an algorithmic worker
            n_w = np.sum(data[i][0:w+1] != -1)
            n_pos = np.sum(data[i][0:w+1] == 1)
            n_neg = np.sum(data[i][0:w+1] == 0)

            maj = 0
            if n_pos == n_neg and n_pos != 0:
                # tie results in switch
                maj = (prev + 1)%2
            elif n_pos > n_w/2:
                maj = 1
            if prev != maj:
                if (maj == 1 and pos_switch) or (maj == 0 and neg_switch):
                    switches[i][w] = 1
            prev = maj
            majority[i][w] = maj
    
    return np.sum(np.logical_and(np.sum(switches,axis=1), np.sum(data,axis=1) != -1*len(data[0])))   


     

"""
    At a given point of time, the required number of switches (i.e., majority concensus flips) 
    to get to the ground truth status.
"""
def gt_switch(data,slist,pos_switch=True,neg_switch=True):
    if len(data[0]) < 1:
        # assuming all clean vector
        return np.sum(slist==1)
    votes = np.zeros((len(data)))
    for i in range(len(data)):
        prev = 0
        for w in range(0,len(data[0])):
            # the first worker is compared with an algorithmic worker
            n_w = np.sum(data[i][0:w+1] != -1)
            n_pos = np.sum(data[i][0:w+1] == 1)
            n_neg = np.sum(data[i][0:w+1] == 0)
            if True:
                maj = 0#-1
                if np.sum(data[i][0:w+1] == 1) > n_w/2:
                    maj = 1
                if n_pos == n_neg and n_pos != 0:
                    # tie results in switch
                    maj = (prev + 1)%2
                prev = maj
                votes[i] = maj 
    
    if pos_switch and not neg_switch:
        return np.sum(np.logical_and(np.logical_xor(votes,np.array(slist)), np.array(slist) == 1))
    elif not pos_switch and neg_switch:
        return np.sum(np.logical_and(np.logical_xor(votes,np.array(slist)), np.array(slist) == 0))
    else:
        # assuming all clean pairs initially for non-reviewed pairs
        return np.sum(np.logical_xor(votes,np.array(slist)))




"""
    If majority decreases, then use neg_adj for FPs; otherwise, use pos_adj for FNs.
"""
def vRemainSwitch2(data):
    n_worker = len(data[0])
    est = vNominal(data)
    thresh = np.max([vNominal(data[:,:n_worker/2]), vNominal(data[:,:n_worker/4]), vNominal(data[:,:n_worker/4*3]) ])
    pos_adj = 0
    neg_adj = 0
    if est - thresh < 0:
        neg_adj = max(0,remain_switch(data,pos_switch=False,neg_switch=True) - sNominal(data,pos_switch=False,neg_switch=True))
    else:
        pos_adj = max(0,remain_switch(data,pos_switch=True,neg_switch=False) - sNominal(data,pos_switch=True,neg_switch=False))
    return max(0,est + pos_adj - neg_adj)

   
"""
    Switch estimator, estimate how many majority concensus would flip.
    We assume that the concensus will eventually converge with enough number of votes,
    as workers are better than random guesser and sample from all possible items.
""" 
def remain_switch(data,corrected=False,pos_switch=True,neg_switch=True):
    data_subset = copy.deepcopy(data)
    majority = np.zeros((len(data_subset),len(data_subset[0])))
    switches = np.zeros((len(data_subset),len(data_subset[0])))
    for i in range(len(data_subset)):
        prev = 0
        for w in range(0,len(data_subset[0])):
            # the first worker is compared with an algorithmic worker
            n_w = np.sum(data[i][0:w+1] != -1)
            n_pos = np.sum(data[i][0:w+1] == 1)
            n_neg = np.sum(data[i][0:w+1] == 0)

            maj = 0
            if n_pos == n_neg and n_pos != 0:
                # tie results in switch
                maj = (prev + 1)%2
            elif n_pos > n_w/2:
                maj = 1
            if prev != maj:
                if (maj == 1 and pos_switch) or (maj == 0 and neg_switch):
                    switches[i][w] = 1
            prev = maj
            majority[i][w] = maj

    n_worker = np.sum(data_subset != -1, axis=1)
    n_all = n_worker
    data_subset[data_subset == -1] = 0
    
    histogram = n_worker 
    n = float(np.sum(n_worker))
    n_bar = float(np.mean(n_worker))
    v_bar = float(np.var(n_worker))
    d = np.sum(np.logical_and(np.sum(switches,axis=1), n_all != 0))   
    if n == 0:
        return d
    
    f1 = 0.
    for i in range(len(switches)):
        if n_worker[i] == 0:
            continue
        for k in range(len(switches[0])):
            j = len(switches[0]) -1 - k
            if data[i][j] == -1:
                continue
            elif switches[i][j] == 1:
                f1 += 1
                break
            else:
                break
    # remove no-ops
    for i in range(len(switches)):
        switch_idx= np.where(switches[i,:]==1)[0]
        if len(switch_idx) > 0:
            n -= np.sum(data[i,:np.amin(switch_idx)] != -1)
        elif len(switch_idx) == 0:
            n -= np.sum(data[i,:] != -1)
    if n == 0:
        return d
    c_hat = 1 - f1/n
    c_hat = max(0,c_hat)
    gamma = v_bar/n_bar
    if c_hat == 0.:
        return d

    if corrected:
        est = d/c_hat + n*(1-c_hat)/c_hat*gamma
    else:
        est = d/c_hat

    return est
