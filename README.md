## A simple implementation of CNN using NumPy

One of the assignments from the deep learning course https://dlcourse.ai served as the foundation. All layers, metrics and optimizers are implemented using NuMpy. 

### Dataset
The model is trained on The Street View House Numbers (SVHN) Dataset (http://ufldl.stanford.edu/housenumbers). SVHN is obtained from house numbers in Google Street View images. Model should distinguish 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10. 

### Training process.
As the project's sole objective is the correct implementation of the neural network, there is a failure to attain a commendable metric on the test data in this work. The quality of the algorithm is assessed through overfitting on a diminutive subset of the training data. 100% accuracy on this subset is counted as the successful implementation. 

### Results
The best results achieved with MomentumSGD optimizer, learning rate equal to 1e-2 batch size equal to 32. Number of epochs was set to 50 but model showed overfit of 100% after 20 epochs.