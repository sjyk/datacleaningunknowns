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
from estimator import chao92, sChao92, nominal, vNominal, sNominal, remain_switch, gt_switch, extrapolation, extrapolation2, extrapolation3,vRemainSwitch, vRemainSwitch2, minTasks, sampleCoverage, minTasksToCleanAll, extrapolateFromSample
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



#address dataset
d,gt,prec,rec = loadAddress()
task_sol = pickle.load( open('dataset/addr_solution.p','rb') )
slist = task_sol.values()
min_tasks5 = minTasksToCleanAll(d)
obvious_err = 0 # no priotization
n_worker = 1000
scale = 100
init = 100
n_rep = 10
priotization = True
logscale = False

(X,Y,GT) = holdoutRealWorkers(d,gt_list,range(init,n_worker,scale),estimators,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/addr_results_1.p','wb'))
#(X,Y,GT) = pickle.load(open('dataset/addr_results_1.p','rb'))
plotY1Y2((X,Y,GT),
                        legend=legend,
                        legend_gt=legend_gt,
                        xaxis="Tasks",
                        yaxis="# Total Error",
                        ymax=250,
                        vertical_y=230,
                        loc='upper right',
                        title='(a) Total Error', 
                        filename="plot/addr_mostly_hard_all.png",
                        min_tasks5=min_tasks5,
                        )

#positive switch estimation
(X,Y,GT) = holdoutRealWorkers(d,gt_list2a,range(init,n_worker,scale),estimators2a,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/addr_results_3.p','wb'))
#(X,Y,GT) = pickle.load(open('dataset/addr_results_3.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend2a,    
                        legend_gt = legend_gt2,    
                        xaxis="Tasks",
                        yaxis="# Remaining Switches",
                        ymax=140,
                        title='(b) Remaining Positive Switches', 
                        filename="plot/addr_mostly_hard_pos_switch.png",
                        )

#negative estimation
(X,Y,GT) = holdoutRealWorkers(d,gt_list2b,range(init,n_worker,scale),estimators2b,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/rest2_results_4.p','wb'))
#(X,Y,GT) = pickle.load(open('dataset/rest2_results_4.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend2b,    
                        legend_gt = legend_gt2,    
                        xaxis="Tasks",
                        yaxis="# Remaining Switches",
                        ymax=140,
                        title='(c) Remaining Negative Switches', 
                        filename="plot/addr_mostly_hard_neg_switch.png",
                        )
