

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
conf_5= confusion_matrix(y_test,y_test_pred_5)


print('{0:0.02f}'.format(np.mean(y_test_pred==y_test)*100), 'of test examples classified correctly for k=1 Neighbour(s).')
print('Confusion Matrix for k=1 Neighbour(s)')
print(conf_1)
print('{0:0.02f}'.format(np.mean(y_test_pred_5==y_test)*100), 'of test examples classified correctly for k=5 Neighbour(s).')
print("Confusion Matrix for k=5 Neighbour(s):")
print(conf_5)
pdb.set_trace()
#2.c
from sklearn.model_selection import KFold


def cross_validation(x_train, y_train, knn=1, K=5):
	kf = KFold(n_splits=K, shuffle=True)
	accuracy_all_fold = []
	i = 0
	for train, test in kf.split(x_train):
		dists =  mlBasics.compute_euclidean_distances(x_train[train],x_train[test])
		y_test_pred = mlBasics.predict_labels(dists, y_train[train] , k=knn)
		accuracy = np.mean(y_test_pred==y_train[test])*100
		accuracy_all_fold.append(accuracy)
		print('K= {1} Fold {0} Accuracy {2:.2f}'.format(knn, i, accuracy))
		i += 1
	return accuracy_all_fold

accuracy_cv = []
accuracy_cv_avg = []
K = 5
knn_ = 16
for k in range(1,knn_+1):
	accuracy_all_fold = cross_validation(x_train_sample, y_train_sample, knn=k, K=K)
	accuracy_cv.append(accuracy_all_fold)
	accuracy_cv_avg.append(np.mean(accuracy_all_fold))

best_k = np.argmax(accuracy_cv_avg)+1


from matplotlib import pyplot as plt
plt.subplot(2,1,1)
x_axis = range(1,K+1)
for k, acc_all_fold in enumerate(accuracy_cv):
	plt.plot(x_axis, acc_all_fold, label='K: ' +'{0}'.format(k+1))

plt.xlabel('fold')
plt.ylabel('accuracy')
plt.legend(loc='upper right', fontsize=8)
plt.savefig('acc_k_vs_fold_.png')
plt.cla()
plt.clf()
x_axis1 = range(1,knn_+1)
plt.plot(x_axis1, accuracy_cv_avg)
plt.xlabel('k as in k-nn')
plt.ylabel('average accuracy')
plt.savefig('k_vs_avg_acc_.png')


#2.d
dists = mlBasics.compute_euclidean_distances(X_train, X_test)
y_test_pred_all = mlBasics.predict_labels(dists, y_train)
print('{0:0.02f}'.format( np.mean(y_test_pred_all==y_test)*100), 'of test examples classified correctly for k=1 Neighbour(s).')

print('Compute Distances')
dists =  mlBasics.compute_euclidean_distances(X_train, X_test)
y_test_best_k = mlBasics.predict_labels(dists, y_train, k=best_k)
print('Accuracy on Test set with best k ={0}; {1:0.02f}'.format(best_k,np.mean(y_test_best_k==y_train)*100))