# rsna_challenge
For our project working with the Kaggle RSNA pneumonia dataset of chest x-rays to detect signs of pneumonia

The goal was to draw boxes that closely match those drawn by radiologists of areas of the X-rays that indicate pneumonia. I took advantage of transfer learning using the popular densely connected network architechture DenseNet121 within Keras with a Tensorflow backend. The classifiers used were SVM, decision tree, random forest, Adaboost, Naive Bayes, nearest neighbours and neural net from SKLearn and were compared on the overlap of the predicted box and the radiologists box, divided by the total area of both (called intersection over union, IOU). This score was chosen because this dataset is highly imblanced, having significanly more area of the images not indicating pneumonia than those that indicate, resulting in the conventional precision score being very misleading since an algorithm that always returns false would have a high (>90%) precision, but a low recall and IOU.

# installation
Download the dataset from Kaggle (https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) and extract to ./input
