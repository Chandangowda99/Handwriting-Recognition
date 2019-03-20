# Handwriting-Recognition
Author: Chandan B.
Import libraries and dataset from Scikit-Learn
In [2]:
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import load_digits

digits = load_digits()
Analyze a sample image
In [3]:
import pylab as pl 
pl.gray() 
pl.matshow(digits.images[0]) 
pl.show()
<matplotlib.figure.Figure at 0x8fab828>

Analyze image pixels
Each element represents the pixel of our greyscale image. The value ranges from 0 to 255 for an 8 bit pixel.
In [4]:
digits.images[0]
Out[4]:
array([[  0.,   0.,   5.,  13.,   9.,   1.,   0.,   0.],
       [  0.,   0.,  13.,  15.,  10.,  15.,   5.,   0.],
       [  0.,   3.,  15.,   2.,   0.,  11.,   8.,   0.],
       [  0.,   4.,  12.,   0.,   0.,   8.,   8.,   0.],
       [  0.,   5.,   8.,   0.,   0.,   9.,   8.,   0.],
       [  0.,   4.,  11.,   0.,   1.,  12.,   7.,   0.],
       [  0.,   2.,  14.,   5.,  10.,  12.,   0.,   0.],
       [  0.,   0.,   6.,  13.,  10.,   0.,   0.,   0.]])
Visualize first 15 images
In [5]:
images_and_labels = list(zip(digits.images, digits.target))
plt.figure(figsize=(5,5))
for index, (image, label) in enumerate(images_and_labels[:15]):
    plt.subplot(3, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('%i' % label)

In [6]:
import random
from sklearn import ensemble

#Define variables
n_samples = len(digits.images)
x = digits.images.reshape((n_samples, -1))
y = digits.target

#Create random indices 
sample_index=random.sample(range(len(x)),len(x)/5) #20-80
valid_index=[i for i in range(len(x)) if i not in sample_index]

#Sample and validation images
sample_images=[x[i] for i in sample_index]
valid_images=[x[i] for i in valid_index]

#Sample and validation targets
sample_target=[y[i] for i in sample_index]
valid_target=[y[i] for i in valid_index]

#Using the Random Forest Classifier
classifier = ensemble.RandomForestClassifier()

#Fit model with sample data
classifier.fit(sample_images, sample_target)

#Attempt to predict validation data
score=classifier.score(valid_images, valid_target)
print 'Random Tree Classifier:\n' 
print 'Score\t'+str(score)
Random Tree Classifier:

Score	0.879694019471
In [7]:
i=150

pl.gray() 
pl.matshow(digits.images[i]) 
pl.show() 
classifier.predict(x[i])
<matplotlib.figure.Figure at 0xa8702e8>

C:\Users\mgalarnyk\AppData\Local\Continuum\Anaconda2\lib\site-packages\sklearn\utils\validation.py:386: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and willraise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
  DeprecationWarning)
Out[7]:
array([0])
