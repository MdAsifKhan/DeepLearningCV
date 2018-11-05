

from load_mnist import * 
import hw1_knn  as mlBasics  
import numpy as np 
import pdb
  

# Load data - ALL CLASSES
#2.a
X_train, y_train = load_mnist('training')
X_test, y_test = load_mnist('testing')

# 2.b
label_sample_idx = {}
for idx, label in enumerate(y_train):
	if label in label_sample_idx:
		label_sample_idx[label].append(idx)
	else:
		label_sample_idx[label] = [idx]

sample_idx = np.empty(1000, dtype='int')
i, j = 0, 100
for label in label_sample_idx:
	sample_idx[i:j] = np.random.choice(label_sample_idx[label], size=100)
	i, j = j, j+100

x_train_sample, y_train_sample = X_train[sample_idx], y_train[sample_idx]

# Reshape images
x_train_sample = np.reshape(x_train_sample, (x_train_sample.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))


print('Compute Distances')
dists =  mlBasics.compute_euclidean_distances(x_train_sample,X_test)

print('For k=1 Neighbour')
y_test_pred = mlBasics.predict_labels(dists, y_train_sample, k=1)

print('For k=5 Neighbours')
y_test_pred_5 = mlBasics.predict_labels(dists, y_train_sample, k=5)


from sklearn.metrics import confusion_matrix

conf_1 = confusion_matrix(y_test,y_test_pred)
conf_5= confusion_matrix(y_test_all,y_test_pred_5)


print('{0:0.02f}'.format(np.mean(y_test_pred==y_test)*100), 'of test examples classified correctly for k=1 Neighbour(s).')
print('Confusion Matrix for k=1 Neighbour(s)')
print(conf_1)
print('{0:0.02f}'.format(np.mean(y_test_pred_5==y_test)*100), 'of test examples classified correctly for k=5 Neighbour(s).')
print("Confusion Matrix for k=5 Neighbour(s):")
print(conf_5)

#2.c
from sklearn.model_selection import KFold


def cross_validation(x_train, knn=1, K=5):
	kf = KFold(n_splits=K)
	accuracy_all_fold = []
	for i, train, test in enumerate(kf.split(x_train)):
    	dists =  mlBasics.compute_euclidean_distances(x_train_sample[train],x_train_sample[test])
		y_test_pred = mlBasics.predict_labels(dists,y_train_sample[train] , k=knn)
	    accuracy = '{0:0.02f}'.format(np.mean(y_test_pred==y_train_sample[test])*100)
	    accuracy_all_fold.append(accuracy)
	    return accuracy_all_fold

accuracy_cv = []
accuracy_cv_avg = []
K = 5
for k in range(1,16):
	accuracy_all_fold = cross_validation(x_train, knn=k, K=K)
	accuracy_cv.append(accuracy_all_fold)
	accuracy_cv_avg.append(np.mean(accuracy_all_fold))
best_k = np.argmax(accuracy_cv_avg)


from matplotlib import pyplot as plt
plt.subplot(2,1,1)
for k, acc_all_fold in enumerate(accuracy_cv):
	plt.plot(range(0,K),acc_all_fold, label='K '+str(k))

plt.xlabel('fold')
plt.ylabel('accuracy')
plt.legend('upper right')

plt.subplot(2,1,2)
plt.plot(range(1,16), accuracy_cv_avg)
plt.xlabel('k as in k-nn')
plt.ylabel('accuracy')
plt.legend('upper right')

plt.show()


#2.d
dists = mlBasics.compute_euclidean_distances(X_train,X_test)
y_test_pred_all = mlBasics.predict_labels(dists, y_train)
print('{0:0.02f}'.format( np.mean(y_test_pred_1==y_test_all)*100), 'of test examples classified correctly for k=1 Neighbour(s).')

print('Compute Distances')
dists =  mlBasics.compute_euclidean_distances(x_train_sample,X_test)
y_test_best_k = mlBasics.predict_labels(dists, y_train_sample, k=best_k)
print('Accuracy on Test set with best k ={0}; {1:0.02f}'.format(best_k,np.mean(y_test_best_k==y_train_sample)*100))