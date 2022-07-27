import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

data = pd.read_csv('heart1.csv')

model = BayesianNetwork([('age', 'heartdisease'), ('sex', 'heartdisease'), ('exang', 'heartdisease'),
                         ('cp', 'heartdisease'), ('restecg', 'heartdisease'), ('chol', 'heartdisease')])
model.fit(data, estimator=MaximumLikelihoodEstimator)

infer = VariableElimination(model)

q1 = infer.query(variables=['heartdisease'], evidence={'restecg': 1})
print(q1)
q2 = infer.query(variables=['heartdisease'], evidence={'cp': 2})
print(q2)