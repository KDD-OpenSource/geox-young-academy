from sklearn import svm
from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Note: Functions should go on top, but to group the code based on the sub-tasks, code and functions are mixed here

def show_raw_image(img):
	
    img2 = np.log(img[[2,1,0],:,:])
    
    img2[0,:,:] = img2[0,:,:].copy() * 1.05303 + -6.32792
    img2[1,:,:] = img2[1,:,:].copy() * 1.74001 + -10.8407
    img2[2,:,:] = img2[2,:,:].copy() * 1.20697 + -6.73016
	
    img2 = np.clip(img2 / 6 + 0.5, 0.0, 1.0)

    plt.imshow(np.transpose(img2, (1, 2, 0)))
    plt.show()


def compute_mean_PCs(X):
    mean = np.mean(X, 0) # Second parameter is the axis along which we want to average (0 == across instances)
    mean_free = X-mean

    vars_per_img = X.shape[1] * X.shape[2] * X.shape[3]
    num_imgs = mean_free.shape[0]

    # Flatten data in to vectors
    mean_free_vectorized = np.reshape(mean_free, (num_imgs, vars_per_img))

    # Increase this to speed up debugging
    covar_subsampling = 2

    # Accumulate covar matrix
    covar = np.zeros((vars_per_img, vars_per_img))
    print("Image: 0")
    for i in range(0, num_imgs, covar_subsampling):
        print("\rImage: {}".format(i))
        covar += np.outer(mean_free_vectorized[i,:], mean_free_vectorized[i,:])

    covar /= num_imgs/covar_subsampling

    eig_val, eig_vec = np.linalg.eig(covar)

    # Sort by importance
    idx = np.argsort(eig_val)[::-1]
    eig_vec = eig_vec[:,idx]
    eig_val = eig_val[idx]

    # Reshape data back into images. Note that eig_vec is the transpose of what you might expect it to be.
    principal_components = np.transpose(eig_vec, (1,0)).reshape((vars_per_img, X.shape[1], X.shape[2], X.shape[3]))

    return mean, principal_components, eig_val



def show_first_principal_components(pcs):
	
    f, axarr = plt.subplots(8,8)
    for i in range(0,8):
        for j in range(0,8):
            img2 = pcs[i*8+j,[2,1,0],:,:]
            img2 = np.clip(img2 * 10 + 0.5, 0.0, 1.0)
            axarr[i,j].imshow(np.transpose(img2, (1, 2, 0)))

    plt.show()

def compute_features(X, mean, principal_components, count):
    X_mean_free = X - mean
    features = np.zeros((X.shape[0], count))
    for i in range(0, X.shape[0]):
        for j in range(0, count):
			# Note: The [i,:,:,:] is being very explicit here. [i] would also work.
            features[i,j] = X_mean_free[i,:,:,:].flatten().dot(principal_components[j,:,:,:].flatten())
    return features
    
def reconstruct_image(feature, mean, principal_components):
    reconstruction = np.copy(mean)
    for i in range(0, feature.shape[0]):
        reconstruction += feature[i] * principal_components[i,:,:,:]
    return reconstruction    
    

def testAccuracy(classifier, test_X, test_Y):
	inferred_Y = classifier.predict(test_X)
	return np.mean(test_Y == inferred_Y)

training_data = np.load('data/s2_training_data.npz')

train_X = training_data['data']
train_Y = training_data['labels']

# Take the logarithm
train_X = np.log(train_X)

### Task 1

(mean, principal_components, eig_val) = compute_mean_PCs(train_X)

train_features = compute_features(train_X, mean, principal_components, 16)

### Task 2

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(train_features, train_Y)

### Task 3

print("training accuracy = {}".format(testAccuracy(clf, train_features, train_Y)))


### Task 4

testing_data = np.load('data/s2_testing_data.npz')
test_X = testing_data['data']
test_Y = testing_data['labels']
test_X = np.log(test_X)

test_features = compute_features(test_X, mean, principal_components, 16)

print("test accuracy = {}".format(testAccuracy(clf, test_features, test_Y)))


# Lets test some descriptor sizes:
for i in range(0, 10):
	size = 1 << i
	print("Running with size = {}".format(size))
	train_features = compute_features(train_X, mean, principal_components, size)
	clf = svm.SVC(gamma=0.001, C=100.)
	clf.fit(train_features, train_Y)
	print("training accuracy = {}".format(testAccuracy(clf, train_features, train_Y)))
	test_features = compute_features(test_X, mean, principal_components, size)
	print("test accuracy = {}".format(testAccuracy(clf, test_features, test_Y)))
	


### Task 5


train_features = compute_features(train_X, mean, principal_components, 64)
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(train_features, train_Y)


application_data = np.load('data/s2_application_data.npz')
application_X = application_data['data']+1e-2

show_raw_image(application_X)
application_X = np.log(application_X)

application_labels = np.zeros((application_X.shape[1]-15, application_X.shape[2]-15))
for y in range(0, application_labels.shape[1]):
    print(y)
    for x in range(0, application_labels.shape[0]):
        crop = np.zeros((1,4,15,15))
        crop[0,:,:,:] = application_X[:,y:y+15,x:x+15]
        features = compute_features(crop, mean, principal_components, 64)
        prediction = clf.predict(features)
        application_labels[y,x] = prediction[0]
        


plt.imshow(application_labels)
plt.show()



