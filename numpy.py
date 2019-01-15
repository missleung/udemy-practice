#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 21:43:11 2018

@author: lisa
"""
import os
os.getcwd()
os.chdir("/Users/lisa/Basic python")

import numpy as np
A = np.array([1,2,3])
A[1]

import pandas as ps
B = ps.Series([2,3,4])
B[1]

I=A*B

dataset = pd.read_csv('/Users/lisa/Neural Network (Udemy)/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 1 - Artificial Neural Networks (ANN)/Section 2 - Part 1 - ANN/Artificial_Neural_Networks/Churn_Modelling.csv')
print(type(dataset))

dataset.info()
dataset.head()

list(dataset.columns)

#Change names
dataset=dataset.rename(columns={'CustomerId':'CustID', 'CreditScore':'Credit'})

C= ps.DataFrame([A,B],index=['a','b'], columns=['aa','bb','cc'])
C.loc['a']['bb']
C.loc['a'][C.loc['a']==1]
type(C.loc['a'])


D = C.values()