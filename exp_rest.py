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
from estimator import chao92, sChao92, nominal, vNominal,  sNominal, remain_switch, gt_switch, extrapolation, extrapolation2, extrapolation3,vRemainSwitch, vRemainSwitch2, minTasks,  minTasksToCleanAll, extrapolateFromSample
from dataload import simulatedData, loadRestaurant2, loadProduct, loadRestaurantExtSample, loadAddress
import pickle
from simulation import plotMulti, plotY1Y2, holdoutRealWorkers



##############################################
########### real-world datasets ##############
##############################################

#====total error====
estimators = [lambda x: extrapolateFromSample(x,slist,0.05)+obvious_err,lambda x:vNominal(x)+obvious_err,  lambda x: sChao92(x,shift=1)+obvious_err, lambda x: vRemainSwitch2(x)+obvious_err]
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


#restaurant dataset
n_worker = 1400
scale = 100
init = 100
priotization = True
logscale = False
n_rep=10

d,gt,prec,rec = loadRestaurant2(['dataset/good_worker/restaurant_additional.csv','dataset/good_worker/restaurant_new.csv'],priotization=priotization)
#['dataset/restaurant_new.csv','dataset/restaurant_new2.csv'],priotization=True)
min_tasks5 = minTasksToCleanAll(d)
pair_solution = pickle.load( open('dataset/pair_solution.p','rb') )
# Pair solution ground truth
slist = pair_solution.values()
print 'restaurant data (hueristic pairs):',len(slist),numpy.sum(numpy.array(slist) == 1)
easy_pair_solution = pickle.load( open('dataset/easy_pair_solution.p','rb') )
easy_slist = easy_pair_solution.values()
print 'restaurant data (easy pairs):',len(easy_slist),numpy.sum(numpy.array(easy_slist) == 1)
obvious_err = numpy.sum(numpy.array(easy_slist) == 1)

(X,Y,GT) = holdoutRealWorkers(d,gt_list,range(init,n_worker,scale),estimators,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/rest2_results_1.p','wb'))
#(X,Y,GT) = pickle.load(open('dataset/rest2_results_1.p','rb'))
plotY1Y2((X,Y,GT),
                        legend=legend,
                        legend_gt=legend_gt,
                        xaxis="Tasks",
                        yaxis="# Total Error",
                        ymax=200,
                        xmin=200,
                        loc='lower right',
                        title='(a) Total Error', 
                        filename="plot/rest2_mostly_hard_all.png",
                        min_tasks5=min_tasks5,
                        vertical_y=70
                        )

#positive switch estimation
print 'switch estimation'
(X,Y,GT) = holdoutRealWorkers(d,gt_list2a,range(init,n_worker,scale),estimators2a,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/rest2_results_3.p','wb'))
#(X,Y,GT) = pickle.load(open('dataset/rest2_results_3.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend2a,    
                        legend_gt = legend_gt2,    
                        xaxis="Tasks",
                        yaxis="# Remaining Switches",
                        ymax=80,
                        xmin=200,
                        title='(b) Remaining Positive Switches', 
                        filename="plot/rest2_mostly_hard_pos_switch.png",
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
                        ymax=80,
                        xmin=200,
                        title='(c) Remaining Negative Switches', 
                        filename="plot/rest2_mostly_hard_neg_switch.png",
                        )
