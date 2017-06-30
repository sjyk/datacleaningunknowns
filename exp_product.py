#!/usr/bin/env python
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy
import scipy
import csv
import math
import random
import simplejson
from estimator import chao92, sChao92, nominal, vNominal, sNominal, remain_switch, gt_switch, extrapolation, extrapolation2, extrapolation3,vRemainSwitch, vRemainSwitch2, extrapolateFromSample, sampleCoverage, minTasks, minTasksToCleanAll
from datagen import generateDist, generateDataset, generateWeightedDataset, shuffleList
from dataload import simulatedData, loadCrowdFlowerData, loadRestaurant2, loadProduct, loadRestaurantExtSample, loadAddress
import pickle
from simulation import plotMulti, plotY1Y2, holdoutRealWorkers


##############################################
########### real-world datasets ##############
##############################################

#====total error====
estimators = [lambda x: extrapolateFromSample(x,slist,0.05)+obvious_err,lambda x:vNominal(x)+obvious_err,  lambda x: sChao92(x,shift=1)+obvious_err, lambda x: vRemainSwitch2(x)+obvious_err]
#estimators = [lambda x: vNominal(x) + obvious_err, lambda x: chao92(x)+obvious_err]
legend = ["EXTRAPOL","VOTING","V-CHAO","SWITCH"]
gt_list = [lambda x: gt+obvious_err,lambda x: gt+obvious_err, lambda x: gt+obvious_err, lambda x: gt+obvious_err]
legend_gt=["Ground Truth"]

#====switch==== 
estimators2 = [lambda x: remain_switch(x) - sNominal(x)]
estimators2a = [lambda x: remain_switch(x,neg_switch=False) - sNominal(x,neg_switch=False)]
estimators2b = [lambda x: remain_switch(x,pos_switch=False) - sNominal(x,pos_switch=False)]
gt_list2 = [lambda x: gt_switch(x,slist)]
gt_list2a = [lambda x: gt_switch(x,slist,neg_switch=False)]
gt_list2b = [lambda x: gt_switch(x,slist,pos_switch=False)]
legend2 = ["REMAIN-SWITCH"]
legend2a = ["REMAIN-SWITCH (+)"]
legend2b = ["REMAIN-SWITCH (-)"]
legend_gt2 = ["Ground Truth"]



#Product dataset
print 'loading product data'
d,gt,prec,rec = loadProduct(['dataset/jn_heur/jn_heur_products.csv'])
pair_solution = pickle.load( open('dataset/jn_heur/pair_solution.p','rb'))
min_tasks5 = minTasksToCleanAll(d)
print 'loaded product data'
slist = pair_solution.values()
print 'product data (hueristic pairs):',len(slist),numpy.sum(numpy.array(slist) == 1)
easy_pair_solution = pickle.load( open('dataset/jn_heur/easy_pair_solution.p','rb'))
easy_slist = easy_pair_solution.values()
print 'product data (easy pairs):',len(easy_slist),numpy.sum(numpy.array(easy_slist) == 1)
obvious_err = numpy.sum(numpy.array(easy_slist) == 1)
n_worker = 4800
scale = 400
init = 400
priotization = True
logscale = False
n_rep = 10

(X,Y,GT) = holdoutRealWorkers(d,gt_list,range(init,n_worker,scale),estimators,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/product_results_1b.p','wb'))
#(X,Y,GT) = pickle.load(open('dataset/product_results_1b.p','rb'))
plotY1Y2((X,Y,GT),
                        legend=legend,
                        legend_gt=legend_gt,
                        xaxis="Tasks",
                        loc='lower right',
                        title="(a) Total Error",
                        yaxis='# Total Error', 
                        ymax=1600,
                        filename="plot/product_mostly_hard_all.png",
                        min_tasks5=min_tasks5,
                        vertical_y=600
                        )

#positive switch estimation
(X,Y,GT) = holdoutRealWorkers(d,gt_list2a,range(init,n_worker,scale),estimators2a,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/product_results_3.p','wb'))
#(X,Y,GT) = pickle.load(open('dataset/product_results_3.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend2a,    
                        legend_gt = legend_gt2,    
                        xaxis="Tasks",
                        ymax=1200,
                        yaxis="# Remaining Switches",
                        title='(b) Remaining Positive Switches', 
                        filename="plot/product_mostly_hard_pos_switch.png",
                        )

#negative estimation
(X,Y,GT) = holdoutRealWorkers(d,gt_list2b,range(init,n_worker,scale),estimators2b,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/product_results_4.p','wb'))
#(X,Y,GT) = pickle.load(open('dataset/product_results_4.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend2b,    
                        legend_gt = legend_gt2,    
                        xaxis="Tasks",
                        ymax=1200,
                        yaxis="# Remaining Switches",
                        title='(c) Remaining Negative Switches', 
                        filename="plot/product_mostly_hard_neg_switch.png",
                        )
