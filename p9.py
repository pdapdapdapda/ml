# Write a program to construct a Bayesian network considering medical data. Use this model to demonstrate the diagnosis of heart patients using standard Heart Disease Data Set. You can use Java/Python ML library classes/API
# OK
import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

# read Cleveland Heart Disease data
heartDisease = pd.read_csv('heart1.csv')
heartDisease = heartDisease.replace('?', np.nan)

# display the data
print('Sample instances from the dataset are given below')
print(heartDisease.head())

# # Creat Model- Bayesian Network
model = BayesianNetwork([('age', 'heartdisease'), ('sex', 'heartdisease'), ('exang', 'heartdisease'),
                         ('cp', 'heartdisease'), ('restecg', 'heartdisease'), ('chol', 'heartdisease')])

# Learning CPDs using Maximum Likelihood Estimators
print('\n Learning CPD using Maximum likelihood estimators')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

# # Inferencing with Bayesian Network
print('\n Inferencing with Bayesian Network:')
HeartDiseasetest_infer = VariableElimination(model)

# # computing the Probability of HeartDisease given restecg
print('\n 1.Probability of HeartDisease given evidence= restecg :1')
q1 = HeartDiseasetest_infer.query(
    variables=['heartdisease'], evidence={'restecg': 1})
print(q1)

# # computing the Probability of HeartDisease given cp
print('\n 2.Probability of HeartDisease given evidence= cp:2 ')
q2 = HeartDiseasetest_infer.query(
    variables=['heartdisease'], evidence={'cp': 2})
print(q2)


##################VIVA##########################
# .fit() basically is used to train model. Latel this model can be used to make predictions
#using .predict()
