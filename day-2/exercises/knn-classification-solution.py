from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Note: Functions should go on top, but to group the code based on the sub-tasks, code and functions are mixed here


np.random.seed(0)

world_is_nice = True

def drawSamples(clusterSize):
	variance = 0.3
	if (not world_is_nice):
		variance = 0.75

	X = np.random.normal([-1.0, -1.0], [variance, variance], size=[clusterSize, 2])
	Y = np.full(shape=clusterSize, fill_value=0)

	X = np.concatenate((X, np.random.normal([-1.0, 1.0], [variance, variance], size=[clusterSize, 2])))
	Y = np.concatenate((Y, np.full(shape=clusterSize, fill_value=1)))

	X = np.concatenate((X, np.random.normal([1.0, -1.0], [variance, variance], size=[clusterSize, 2])))
	Y = np.concatenate((Y, np.full(shape=clusterSize, fill_value=1)))

	X = np.concatenate((X, np.random.normal([1.0, 1.0], [variance, variance], size=[clusterSize, 2])))
	Y = np.concatenate((Y, np.full(shape=clusterSize, fill_value=0)))
	return X, Y


### Task 1

def plotCircles(X, Y):
	# Split the array based on the label
	redCircles = X[Y == 0]
	blueCircles = X[Y == 1]
	# Render red circles first
	plt.plot(redCircles[:,0], redCircles[:,1], 'ro')
	# then blue circles
	plt.plot(blueCircles[:,0], blueCircles[:,1], 'bo')

(train_X,train_Y) = drawSamples(100)
plotCircles(train_X,train_Y)
plt.show()


### Task 2


def plotDecisionBoundaries(train_X, classifier, resolution):
	x_min, x_max = train_X[:, 0].min() - 1, train_X[:, 0].max() + 1
	y_min, y_max = train_X[:, 1].min() - 1, train_X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
		                 np.arange(y_min, y_max, resolution))
	Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]

	Z = Z.reshape(xx.shape)
	plt.pcolormesh(xx, yy, Z, cmap=plt.cm.RdBu)


# "Train" classifier
num_neighbors = 15
clf = neighbors.KNeighborsClassifier(num_neighbors, weights='uniform')
clf.fit(train_X,train_Y)

# Plot data and decision boundaries
plt.figure()
plotDecisionBoundaries(train_X, clf, 0.01)
plotCircles(train_X,train_Y)
plt.show()



### Task 3


def testAccuracy(classifier, test_X, test_Y):
	inferred_Y = classifier.predict(test_X)
	return np.mean(test_Y == inferred_Y)

print("Accuracy = {}".format(testAccuracy(clf, train_X, train_Y)))

### Task 4

world_is_nice = False
(train_X,train_Y) = drawSamples(100)

# "Train" classifier
num_neighbors = 1 # Best performance on training data with k=1
clf = neighbors.KNeighborsClassifier(num_neighbors, weights='uniform')
clf.fit(train_X,train_Y)

# Plot data and decision boundaries
plt.figure()
plotDecisionBoundaries(train_X, clf, 0.01)
plotCircles(train_X,train_Y)
plt.show()


print("Accuracy = {}".format(testAccuracy(clf, train_X, train_Y)))


# Produce test data
(test_X,test_Y) = drawSamples(100)
print("Test accuracy = {}".format(testAccuracy(clf, test_X,test_Y)))


# Lets test different k values 
    # Bonus points: Try a larger dataset
    # (train_X,train_Y) = drawSamples(1000)
    # (test_X,test_Y) = drawSamples(1000)
for k in range(1, 25):
	num_neighbors = k
	clf = neighbors.KNeighborsClassifier(num_neighbors, weights='uniform')
	clf.fit(train_X,train_Y)
	print("Number of neighbors: {}".format(k))
	print("Training accuracy = {}".format(testAccuracy(clf, train_X, train_Y)))
	print("Test accuracy = {}".format(testAccuracy(clf, test_X,test_Y)))




