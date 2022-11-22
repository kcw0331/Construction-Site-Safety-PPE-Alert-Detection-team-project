'''
fashion_cnn_v*.py 결과 csv 파일을 dacon 양식에 맞게 변경하는 코드
'''

import sys
import os

import pandas as pd


columns = ['index', 'label']
types = ['int', 'int']

input_csv = pd.read_csv("results_fashion_mnist.csv").dropna(subset='label')
print(input_csv.dtypes)

# input_csv = pd.read_csv("result.csv")
# print(input_csv)

input_csv = input_csv.astype({'label' : 'int'})
print(input_csv.dtypes)

input_csv['index'] = input_csv['index'] - 1
print(input_csv)

input_csv.to_csv('result.csv',index=None)
