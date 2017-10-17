from sklearn import svm
from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Note: Functions should go on top, but to group the code based on the sub-tasks, code and functions are mixed here

### Task 1

def show_raw_image(img):
	
    img2 = np.log(img[[2,1,0],:,:])
    
    img2[0,:,:] = img2[0,:,:].copy() * 1.05303 + -6.32792
    img2[1,:,:] = img2[1,:,:].copy() * 1.74001 + -10.8407
    img2[2,:,:] = img2[2,:,:].copy() * 1.20697 + -6.73016
	
    img2 = np.clip(img2 / 6 + 0.5, 0.0, 1.0)

    plt.imshow(np.transpose(img2, (1, 2, 0)))
    plt.show()


training_data = np.load('data/s2_training_data.npz')

train_X = training_data['data']
train_Y = training_data['labels']


print("data = {}".format(train_X.shape))
print("labels = {}".format(train_Y.shape))

# Take a look at the first image of each class in turn (each class is 5k images and they are actually sorted by class)
for i in range(0,4):
    show_raw_image(train_X[5000*i])

### Task 2

# Take the logarithm
train_X = np.log(train_X)


### Task 3-7

def compute_mean_PCs(X):
	### Task 3
    # Compute and subtract mean 
    mean = np.mean(X, 0) # Second parameter is the axis along which we want to average (0 == across instances)
    mean_free = X-mean

    vars_per_img = X.shape[1] * X.shape[2] * X.shape[3]
    num_imgs = mean_free.shape[0]

	### Task 4
    # Flatten data in to vectors
    mean_free_vectorized = np.reshape(mean_free, (num_imgs, vars_per_img))

	### Task 5
    # Increase this to speed up debugging
    covar_subsampling = 2

    # Accumulate covar matrix
    covar = np.zeros((vars_per_img, vars_per_img))
    print("Image: 0")
    for i in range(0, num_imgs, covar_subsampling):
        print("\rImage: {}".format(i))
        covar += np.outer(mean_free_vectorized[i,:], mean_free_vectorized[i,:])

    covar /= num_imgs/covar_subsampling

	### Task 6
    eig_val, eig_vec = np.linalg.eig(covar)

    # Sort by importance
    idx = np.argsort(eig_val)[::-1]
    eig_vec = eig_vec[:,idx]
    eig_val = eig_val[idx]

	### Task 7
    # Reshape data back into images. Note that eig_vec is the transpose of what you might expect it to be.
    principal_components = np.transpose(eig_vec, (1,0)).reshape((vars_per_img, X.shape[1], X.shape[2], X.shape[3]))

    return mean, principal_components, eig_val



### Task 8

def show_first_principal_components(pcs):
	
    f, axarr = plt.subplots(8,8)
    for i in range(0,8):
        for j in range(0,8):
            img2 = pcs[i*8+j,[2,1,0],:,:]
            img2 = np.clip(img2 * 10 + 0.5, 0.0, 1.0)
            axarr[i,j].imshow(np.transpose(img2, (1, 2, 0)))

    plt.show()


(mean, principal_components, eig_val) = compute_mean_PCs(train_X)

plt.plot(eig_val)
plt.show()
show_first_principal_components(principal_components)



### Task 9
def compute_features(X, mean, principal_components, count):
    X_mean_free = X - mean
    features = np.zeros((X.shape[0], count))
    for i in range(0, X.shape[0]):
        for j in range(0, count):
			# Note: The [i,:,:,:] is being very explicit here. [i] would also work.
            features[i,j] = X_mean_free[i,:,:,:].flatten().dot(principal_components[j,:,:,:].flatten())
    return features
    
### Task 10
def reconstruct_image(feature, mean, principal_components):
    reconstruction = np.copy(mean)
    for i in range(0, feature.shape[0]):
        reconstruction += feature[i] * principal_components[i,:,:,:]
    return reconstruction    
    

train_features = compute_features(train_X, mean, principal_components, 32)


for i in range(0,4):    
    img = np.concatenate((train_X[5000*i+0,:,:,:], reconstruct_image(train_features[5000*i+0,:], mean, principal_components)), 2);
    img = np.concatenate((img,np.concatenate((train_X[5000*i+1,:,:,:], reconstruct_image(train_features[5000*i+1,:], mean, principal_components)), 2)), 1);
    img = np.concatenate((img,np.concatenate((train_X[5000*i+2,:,:,:], reconstruct_image(train_features[5000*i+2,:], mean, principal_components)), 2)), 1);
    img = np.concatenate((img,np.concatenate((train_X[5000*i+3,:,:,:], reconstruct_image(train_features[5000*i+3,:], mean, principal_components)), 2)), 1);
    img = np.concatenate((img,np.concatenate((train_X[5000*i+4,:,:,:], reconstruct_image(train_features[5000*i+4,:], mean, principal_components)), 2)), 1);
    show_raw_image(np.exp(img))



