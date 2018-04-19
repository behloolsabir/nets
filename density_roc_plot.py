import numpy
from matplotlib import pyplot
import pandas as pd
data = pd.read_csv('probability.csv')
x = data.loc[data.truth==1]['predicted_prob'].tolist()
y = data.loc[data.truth==0]['predicted_prob'].tolist()
bins = numpy.linspace(0.01, 1, 100)
pyplot.hist(x, bins, alpha=0.5, label='Rejected', normed=True)
pyplot.hist(y, bins, alpha=0.5, label='Approved', normed=True)
pyplot.legend(loc='upper right')
# pyplot.show()
pyplot.savefig('density_13000_trim.jpg', dpi=1000)


pyplot.hist(x, bins, alpha=0.5, label='x', density=True, stacked=True, normed=True)



from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

y_true = data.truth
y_probas = data.predicted_prob
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probas, pos_label=1)

# Print ROC curve
plt.plot(fpr,tpr)
plt.show() 

# Print AUC
auc = np.trapz(tpr,fpr)
print('AUC:', auc)
