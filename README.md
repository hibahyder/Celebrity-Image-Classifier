# Celebrity-Image-Classifier

SUMMARY

The chosen model is a Convolutional Neural Network (CNN) implemented using TensorFlow's Keras API. The model architecture consists of the following layers:
1.	Convolutional layer with 32 filters, a kernel size of (3,3), and ReLU activation function.
2.	MaxPooling layer with a pool size of (2,2).
3.	Flatten layer to convert the 2D feature maps to a 1D vector.
4.	Dense layer with 256 neurons and ReLU activation function.
5.	Dropout layer with a dropout rate of 0.5 to prevent overfitting.
6.	Dense layer with 512 neurons and ReLU activation function.
7.	Output layer with 5 neurons (equal to the number of classes) and softmax activation function.
Training Process
1.	Data Loading and Preprocessing:
•	The dataset consists of images of five celebrities: Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli.
•	Images are loaded, resized to (128,128) pixels, and stored in a NumPy array (dataset).
•	Labels are assigned to each celebrity class (0 to 4).
•	The dataset is split into training and testing sets using an 80-20 split.
2.	Model Compilation:
•	The CNN model is compiled with the Adam optimizer, Categorical Crossentropy loss function, and accuracy as the evaluation metric.
3.	One-Hot Encoding:
•	Labels are one-hot encoded to match the categorical nature of the problem.
4.	Model Training:
•	The model is trained for 50 epochs with a batch size of 32.
•	Training progress is monitored, and validation data is used to assess model performance during training.
5.	Model Evaluation:
•	The trained model is evaluated on the test set, and accuracy is reported.
•	Classification report is generated, including precision, recall, and F1-score for each class.
Findings
1. Normalization:
•	The dataset is normalized by scaling pixel values to the range [0, 1].
2. Model Architecture:
•	The CNN architecture is relatively simple, with one convolutional layer followed by max-pooling, flattening, and two dense layers.
•	The model uses dropout to reduce overfitting.
3. Training Performance:
•	Training and validation accuracy are monitored throughout the training process.
•	Model accuracy on the test set is reported after training.
4. Prediction:
•	The model is tested on new images using the preprocess_single_image function.
•	Predictions are made for celebrity classes, and results are printed.

