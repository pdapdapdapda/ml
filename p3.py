import csv
a = []
num_attributes = 6
with open('enjoysport.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    a = list(reader)
hypothesis = ['0' for i in range(num_attributes)]
print("Most specefic hypothesis at the beginning: ")
hypothesis = a[0][:-1]
print(hypothesis)

for i in range(len(a)):
    if a[i][num_attributes] != 'no':
        for j in range(num_attributes):
            if(a[i][j] != hypothesis[j]):
                hypothesis[j] = '?'
            else:
                hypothesis[j] = a[i][j]
print("\nThe Maximally Specific Hypothesis for a given Training Examples :\n")
print(hypothesis)
