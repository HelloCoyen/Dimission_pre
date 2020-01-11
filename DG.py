# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:45:32 2020

@author: wuhaoyu
"""
import pandas as pd
import os

PATH = os.path.dirname(os.path.realpath(__file__))

class DataSet():
    def __init__(self):
        train_df = pd.read_csv(PATH+"\\data\\pfm_train.csv", index_col = "EmployeeNumber")
        test_df = pd.read_csv(PATH+"\\data\\pfm_test.csv", index_col = "EmployeeNumber")
        all_df = pd.concat([train_df, test_df])

#        train_labels = train_df['Attrition']
#        train_factors = train_df.drop(['Attrition'], axis=1)
#        train_factors['BusinessTravel'] = train_factors['BusinessTravel'].astype('category')
