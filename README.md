# DigitRecognizer

[![launch_button](https://user-images.githubusercontent.com/42754056/50173639-ab729b00-02bd-11e9-82fc-d0add39f394f.png)](https://smoijueh-mnist.netlify.com)

Click on the button above to go to the R Notebook. 

Exploratory Data Analysis of the MNIST dataset. Topics: (1) Handwriting Recognition (2) Classification

Final model uses k-nearest neighbor (kNN) to classify the handwritten images. Training instances are stored in a KD tree data structure. This dramatically improved the model's runtime performance. I reduced the dimensionality of the feature space by performing a principal component analysis. Lastly, I optimized for k-value in kNN by performing a k-fold cross validation.
