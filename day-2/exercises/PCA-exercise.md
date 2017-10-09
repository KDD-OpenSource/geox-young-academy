PCA Exercise
===============

Goals
-----

- Implement PCA for dimensionality reduction

Requirements
-----
- numpy
- matplotlib

Exercise
-----

Download the following dataset: [Sentinel 2 hyperspectral](data/s2_training_data.npz)
The dataset contains small patches (15x15 pixels, 4 channels) of Sentinel 2 satellite images. The channels are the RGB channels of the visible spectrum and one near infrared band.

The data can be loaded like this:
```python
training_data = np.load('data/s2_training_data.npz')

train_X = training_data['data']
train_Y = training_data['labels']
```
The labels can be ignored for now. We want to run PCA on the 20000 patches in this dataset. The data is organized as one numpy array of shape (20000, 4, 15, 15) where the first dimension are the instances, the next dimension are the channels, and the latter two are width and height.


### Task 1

Use the following function to look at the (RGB-part) of individual crops:
```python
def show_raw_image(img):
    img2 = np.log(img[[2,1,0],:,:])
    
    img2[0,:,:] = img2[0,:,:].copy() * 1.05303 + -6.32792
    img2[1,:,:] = img2[1,:,:].copy() * 1.74001 + -10.8407
    img2[2,:,:] = img2[2,:,:].copy() * 1.20697 + -6.73016
	
    img2 = np.clip(img2 / 6 + 0.5, 0.0, 1.0)

    plt.imshow(np.transpose(img2, (1, 2, 0)))
    plt.show()
```

The first 5k crops contain forest in the central pixel. The next 5k contain fields/lower vegetation. The next 5k are urban areas and the last 5k are water.

### Task 2

The data train_X contains the raw values which span a large range. To compress them into a more "gaussian" shape, compute the logarithm of them.

### Task 3

This and the following tasks will implement a function compute_mean_PCs(X) which takes image crops and returns a tuple containing the mean image, Eigen images (principal components reshaped into images) and Eigen values. The latter two shall be sorted in decreasing Eigen value order.

```python
def compute_mean_PCs(X):
    vars_per_img = X.shape[1] * X.shape[2] * X.shape[3]
    num_imgs = mean_free.shape[0]

	# todo: stuff

    return mean, principal_components, eig_val
```

Implement code to compute the mean image. Subtract the mean from the input crops "X" to get the mean free crops.

### Task 4

The mean free crops should be a numpy array of shape (20000, 4, 15, 15). Flatten the crops into vectors so that you get a numpy array of shape (20000, 4*15*15). Hint: use "np.reshape".

### Task 5

Compute the covariance matrix. This should be a matrix of size 900x900 (900 = 4*15*15). Start with a zero matrix, loop over all 20000 crops, and for each crop compute the outer product "np.outer(a,b)" of the mean free vectors with themselves. Accumulate (sum) the outer products together and finally divide by the total number of used crops. To speed up development, consider using only every 10th crop.

### Task 6

Compute the Eigen values and vectors of the covariance matrix and sort them by size:
```python
eig_val, eig_vec = np.linalg.eig(covar)

# Sort by importance
idx = np.argsort(eig_val)[::-1]
eig_vec = eig_vec[:,idx]
eig_val = eig_val[idx]
```

### Task 7

Reshape the 900 eigen vectors into the form (900, 4, 15, 15). Note that the matrix eig_vec contains the Eigen vectors in its rows, not columns as you might expect. Return the mean, reshaped Eigen vectors, and Eigen values from the function.

### Task 8

Run compute_mean_PCs on the sattelite data and plot the Eigen values. 
Look at the first 64 principal components (or the RGB part thereof) using the following function:
```python
def show_first_principal_components(pcs):
	
    f, axarr = plt.subplots(8,8)
    for i in range(0,8):
        for j in range(0,8):
            img2 = pcs[i*8+j,[2,1,0],:,:]
            img2 = np.clip(img2 * 10 + 0.5, 0.0, 1.0)
            axarr[i,j].imshow(np.transpose(img2, (1, 2, 0)))

    plt.show()
```

### Task 9

Write a function "compute_features(X, mean, principal_components, count)" that takes crops "X", subtracts the mean, and projects them onto the first "count" principal components. Return a matrix of shape (X.shape[0], count) that contains the coefficients (which we will later on use as features in another exercise). 

### Task 10

Write a function "reconstruct_image(feature, mean, principal_components)" that restores a crop given a feature/coefficient vector.
Use the following code to compare, side by side, original image crops and reconstructions:

```python
train_features = compute_features(train_X, mean, principal_components, 32)

for i in range(0,4):    
    img = np.concatenate((train_X[5000*i,:,:,:], reconstruct_image(train_features[5000*i,:], mean, principal_components)), 2);
    show_raw_image(np.exp(img))
```

