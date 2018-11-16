# DigitRecognizer
Exploratory Data Analysis of the MNIST dataset. Topics: (1) Handwriting Recognition (2) Classification

Final model uses k-nearest neighbor (kNN) to classify the handwritten images. Training instances are stored in a KD tree data structure. This dramatically improved the model's runtime performance. I reduced the dimensionality of the feature space by performing a principal component analysis. Lastly, I optimized for k-value in kNN by performing a k-fold cross validation.

Link to R Notebook: https://smoijueh-mnist.netlify.com
