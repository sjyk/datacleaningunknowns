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
from estimator import chao92, sChao92, nominal, vNominal,  sNominal, remain_switch, gt_switch, extrapolation, extrapolation2, extrapolation3,vRemainSwitch, vRemainSwitch2, minTasks, minTasksToCleanAll,extrapolateFromSample
from datagen import generateDist, generateDataset, generateWeightedDataset, shuffleList
from dataload import simulatedData, loadCrowdFlowerData, loadRestaurant2, loadProduct, loadRestaurantExtSample, loadAddress
import pickle
from simulation import plotMulti, plotY1Y2, holdoutRealWorkers


######################################
###########simulated dataset##########
######################################
logscale=False
dirty = 0.5
n_items = 1000
n_rep = 3


print 'Sensitivity of Total Error Estimation'
estimators_sim = [lambda x:vNominal(x),chao92,lambda x:vRemainSwitch2(x)]
gt_list_sim = [lambda x:gt,lambda x:gt,lambda x:gt]
legend_sim = ['VOTING','Chao92','SWITCH']
legend_gt=["Ground Truth"]
yaxis = 'SRMSE'#'Relative Error %'

rel_err = True
err_skew = False

title = 'Tradeoff: False Positives'
recall = 0.1
n_worker=50
font = 20
Xs = []
Ys = []
GTs = []
for i in range(0,11):
    prec = float(i)/10.
    d,gt,pr,re = simulatedData(items=n_items,recall=recall,precision=prec,error_type=0,err_skew=err_skew)
    slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
    (X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,[n_worker],estimators_sim,rel_err=rel_err,rep=n_rep)
    Xs.append(prec*100)
    Ys.append(Y[0])
    GTs.append(GT[0])

plotY1Y2((Xs,Ys,GTs),
                        legend = legend_sim,
                        xaxis='Precision %',
                        yaxis=yaxis,
                        title=title, 
                        loc='upper right',
                        filename='plot/sim_5a.png',
                        logscale=logscale,
                        rel_err=rel_err,
                        font=font
                        )
title = 'Tradeoff: False Negatives'
precision = 1
Xs = []
Ys = []
GTs = []
for i in range(0,11):
    rec = float(i)/200.
    d,gt,pr,re = simulatedData(items=n_items,recall=rec,precision=precision,error_type=0,err_skew=err_skew)
    slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
    (X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,[n_worker],estimators_sim,rel_err=rel_err,rep=n_rep)
    Xs.append((rec)*100)
    Ys.append(Y[0])
    GTs.append(GT[0])

plotY1Y2((Xs,Ys,GTs),
                        legend = legend_sim,
                        xaxis='Coverage %',
                        yaxis=yaxis,
                        title=title, 
                        loc='upper right',
                        filename='plot/sim_5d.png',
                        logscale=logscale,
                        rel_err=rel_err,
                        font=font
                        )


print 'Sensitivity of Total Error Estimation'
estimators_sim = [vNominal, chao92, lambda x: sChao92(x,shift=1),lambda x: vRemainSwitch2(x)]
gt_list_sim=[lambda x: gt, lambda x: gt, lambda x: gt, lambda x: gt]
legend_sim = ['VOTING','Chao92','V-CHAO(s=1)','SWITCH']
yaxis='Estimate (# Total Error)'

rel_err = False

hir = 0.1 #0.1
lowr = 0.015 #30 items per task
hiq = 0.99 
lowq = 0.9
fnr = 0.1 # 10%? was 1%
fpr = 0.01 # 1%? was 10%

title = 'Perfect Precision'
print title

err_skew = False
error_type = 0
n_worker = 330
scale = 30
init = 30

d,gt,prec,rec = simulatedData(items=n_items,workers=n_worker,dirty=dirty,recall=lowr,precision=1,err_skew=err_skew)
slist = pickle.load( open('dataset/sim_label.p','rb'))
min_tasks = minTasksToCleanAll(d)
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=rel_err,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/7.p','wb'))
#(X,Y,GT) = pickle.load(open('dataset/7.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend_sim,
                        legend_gt = legend_gt,
                        xaxis='Tasks',
                        yaxis=yaxis,
                        ymax=250,
                        title=title, 
                        loc='lower right',
                        filename='plot/sim_7.png',
                        logscale=logscale,
                        rel_err=rel_err,
                        min_tasks2=min_tasks
                        )

fig, ax = plt.subplots(2,3,figsize=(20,8),sharex=True,sharey=True)

n_worker = 500
scale = 50
init = 0

ymax=350
err_skew = False
error_type = 1

title = 'False Positive Errors\n\n(b)'
print title
d,gt,prec,rec = simulatedData(items=n_items,workers=n_worker,dirty=dirty,recall=lowr,precision=hiq, err_skew=err_skew,error_type=error_type)
min_tasks=minTasksToCleanAll(d)#minTasks3(d)
slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=rel_err,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/matrix_b.p','wb'))
#(X,Y,GT) = pickle.load(open('dataset/matrix_b.p','rb'))
plotMulti(ax[0][1], (X,Y,GT),
                        legend = legend_sim,
                        legend_gt = legend_gt,
                        ymax=ymax,
                        yaxis='',
                        title=title, 
                        loc='lower right',
                        filename='plot/sim_00.png',
                        rel_err=rel_err,
                        min_tasks=min_tasks
                        )

title = 'False Negative Errors\n\n(a)'
print title
error_type = 2

d,gt,prec,rec = simulatedData(items=n_items,workers=n_worker,dirty=dirty,recall=lowr,precision=lowq,err_skew=err_skew,error_type=error_type)
slist = pickle.load( open('dataset/sim_label.p','rb'))
min_tasks=minTasksToCleanAll(d) 
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=rel_err,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/matrix_a.p','wb'))
#(X,Y,GT) = pickle.load(open('dataset/matrix_a.p','rb'))
plotMulti(ax[0][0], (X,Y,GT),
                        legend = legend_sim,
                        legend_gt = legend_gt,
                        ymax=ymax,
                        vertical_y=ymax-100,
                        yaxis='Uniform Precision\n\n' + yaxis,
                        title=title, 
                        loc='lower right',
                        filename='plot/sim_01.png',
                        rel_err=rel_err,
                        min_tasks=min_tasks
                        )

title = 'All Errors\n\n(c)'
print title
error_type = 0

d,gt,prec,rec = simulatedData(items=n_items,workers=n_worker,dirty=dirty,recall=lowr,fpr=1-hiq, fnr=1-lowq,err_skew=err_skew,error_type=error_type)
slist = pickle.load( open('dataset/sim_label.p','rb'))
min_tasks=minTasksToCleanAll(d)
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=rel_err,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/matrix_c.p','wb'))
#(X,Y,GT) = pickle.load(open('dataset/matrix_c.p','rb'))
plotMulti(ax[0][2], (X,Y,GT),
                        legend = legend_sim,
                        legend_gt = legend_gt,
                        yaxis='',
                        title=title, 
                        ymax=ymax,
                        loc='lower right',
                        filename='plot/sim_02.png',
                        rel_err=rel_err,
                        min_tasks=min_tasks
                        )


err_skew = True
error_type = 1

title = 'False Positive Errors'
print title
d,gt,prec,rec = simulatedData(items=n_items,workers=n_worker,dirty=dirty,recall=lowr,precision=hiq,err_skew=err_skew,error_type=error_type)
slist = pickle.load( open('dataset/sim_label.p','rb'))
min_tasks=minTasksToCleanAll(d)
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=rel_err,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/matrix_e.p','wb'))
#(X,Y,GT) = pickle.load(open('dataset/matrix_e.p','rb'))
plotMulti(ax[1][1], (X,Y,GT),
                        legend = legend_sim,
                        legend_gt = legend_gt,
                        xaxis='Tasks',
                        yaxis='',
                        title='(e)',
                        ymax=ymax,
                        loc='lower right',
                        filename='plot/sim_10.png',
                        rel_err=rel_err,
                        min_tasks=min_tasks
                        )

title = 'False Negative Errors'
print title
error_type = 2

d,gt,prec,rec = simulatedData(items=n_items,workers=n_worker,dirty=dirty,recall=lowr,precision=lowq,err_skew=err_skew,error_type=error_type)
slist = pickle.load( open('dataset/sim_label.p','rb'))
min_tasks=minTasksToCleanAll(d)
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=rel_err,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/matrix_d.p','wb'))
#(X,Y,GT) = pickle.load(open('dataset/matrix_d.p','rb'))
plotMulti(ax[1][0], (X,Y,GT),
                        legend = legend_sim,
                        legend_gt = legend_gt,
                        xaxis='Tasks',
                        yaxis='Skewed Precision\n\n' + yaxis,
                        vertical_y=ymax-100,
                        title='(d)',
                        ymax=ymax,
                        loc='lower right',
                        filename='plot/sim_11.png',
                        rel_err=rel_err,
                        min_tasks=min_tasks
                        )

title = 'All Errors'
print title
error_type = 0

d,gt,prec,rec = simulatedData(items=n_items,workers=n_worker,dirty=dirty,recall=lowr,fpr=1-hiq,fnr=1-lowq,err_skew=err_skew,error_type=error_type)
slist = pickle.load( open('dataset/sim_label.p','rb'))
min_tasks=minTasksToCleanAll(d)
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=rel_err,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/matrix_f.p','wb'))
#(X,Y,GT) = pickle.load(open('dataset/matrix_f.p','rb'))
plotMulti(ax[1][2], (X,Y,GT),
                        legend = legend_sim,
                        legend_gt = legend_gt,
                        xaxis='Tasks',
                        yaxis='',
                        title='(f)',
                        ymax=ymax,
                        loc='lower right',
                        filename='plot/sim_12.png',
                        rel_err=rel_err,
                        min_tasks=min_tasks
                        )

plt.legend(loc='lower center',bbox_to_anchor = (0,-0.1,1,1),ncol=5,bbox_transform=plt.gcf().transFigure)
plt.savefig('plot/sim_matrix_fnr10_fpr1_1000_100.png',bbox_inches='tight')

