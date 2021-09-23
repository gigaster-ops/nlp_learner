import matplotlib.pyplot as plt
import pandas as pd

path = 'Spam Threats.csv'
column = 'text'

csv = pd.read_csv(path)
print(csv.head())
csv = csv[column]

lens = []
for i in csv:
    lens.append(len(i))

lens = list(set(lens))
plt.hist(lens)
plt.show()