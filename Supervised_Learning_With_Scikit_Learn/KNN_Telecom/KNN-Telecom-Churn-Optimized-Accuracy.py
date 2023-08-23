# KNN Machine Learning example in Python 
# Model optimized on Accuracy

# In this example we wll be predicting customer churn from the "telecom_churn_clean.csv" dataframe.
# This is the optimized model.

# KNN is a Kappa Nearest Neighbors algorithm, kappa represents k or n-th number.
# KNN catagorizes an unknown to the avg of the NN, ie. takes the majority vote of local neighbors.
# Example of prediction in a 3-NN: unknown=u, 1=yes, 2=no, 3=yes, then u=yes.
# Larger K = less complex model (takes the mean of more "votes" in the dataset) = can cause underfitting
# Smaller K = more complex model (takes mean of only one, or a few "votes") = can cause overfitting

# Imports
# pip3 install scikit-learn
# pip3 install pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import pandas as pd


# Loading dataframe
churn_df = pd.read_csv('_datasets/telecom_churn_clean.csv')

# Subsetting columns that we will use for this model.
# Calling a dataframe and passing a list of column names will select/subset those columns.
churn_df = churn_df[['account_length',
                    'total_day_charge',
                    'total_eve_charge',
                    'total_night_charge',
                    'total_intl_charge',
                    'customer_service_calls',
                    'churn']]

# X will contain the features, keep all columns except the target.
# The fastest and most fool-proof way to get X would be just to drop the target, axis=0 is row and axis=1 is column.
# y will contain our target, subsetting for only the target column.
X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values

# Splitting into training and test sets using the imported function "train_test_split()".
# "train_test_split()" takes the features X and the target y.
# The parameter "test_size=" is splitting for 80% train 20% test/validation.
# The parameter "random_state=" is essentially similar to setting a seed. 
# The "stratify=" parameter ensures that the distribution of classes in the original dataset is preserved in both the training and testing sets.
# This is especially useful when dealing with imbalanced datasets, where one class may be significantly underrepresented compared to others.
# For example, if the original dataset has 80% of samples belonging to Class A and 20% belonging to Class B.
# Using "stratify=y" when splitting the dataset will ensure that both the training and testing sets have the same proportion of Class A and Class B samples.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# I know the optimal value for this model is 11.
# Please refer to "KNN_Evaluate_Optimal_Kappa.py" located in the same repository
knn = KNeighborsClassifier(n_neighbors=11)

# Fitting the model to training data.
knn.fit(X_train, y_train)

# Predicting on our model
y_pred = knn.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Calculate the specificity and sensitivity
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)  # Sensitivity and Recall are the same things, just wanted to have that reminder in this test

# Calculate the precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display the confusion matrix and classification report for your model
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Specificity:", specificity)
print("Sensitivity:", sensitivity)
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)

# ----------Terms----------
# Accuracy:                             This metric measures how often the model correctly identifies both positive and negative cases.
# Support:                              Support is the number of instances in each target class. It gives an indication of the prevalence of each class in the dataset.
# F1-score:                             F1-score is the harmonic mean of precision and recall, which gives an overall measure of a classifier's performance. A high F1-score indicates that the classifier performs well in terms of both precision and recall.
# Recall (also known as Sensitivity):   Recall is the proportion of true positive predictions among all actual positive instances. It measures how well the model captures instances of the target class. A high recall score indicates that the model correctly identifies most instances of the target class.
# Precision:                            Precision is the proportion of true positive predictions among all positive predictions. It measures how often the model correctly identifies instances of the target class. A high precision score indicates that the model makes relatively few false positive errors.
# Specificity:                          This metric measures the proportion of true negatives among all actual negative cases. 
# Macro Average (macro avg):            Macro average calculates the average of the precision, recall, and F1-score across all classes, giving equal weight to each class. It is useful when you have imbalanced classes and want to get an overall sense of the model's performance.
# Weighted Average (weighted avg):      Weighted average calculates the average of the precision, recall, and F1-score across all classes, but gives more weight to classes with more instances. It is useful when you have imbalanced classes and want to get an overall sense of the model's performance, but want to give more importance to the larger classes.

# ----------Terminal Output----------
# Confusion Matrix:
# [[560  10]
# [ 76  21]]
#Classification Report:
#               precision    recall  f1-score   support
#
#           0       0.88      0.98      0.93       570
#           1       0.68      0.22      0.33        97

#    accuracy                           0.87       667
#   macro avg       0.78      0.60      0.63       667
# weighted avg       0.85      0.87      0.84       667

# Specificity: 0.9824561403508771
# Sensitivity: 0.21649484536082475
# Precision: 0.6774193548387096
# Recall: 0.21649484536082475
# Accuracy: 0.8710644677661169

# ----------Conclusion----------
# My personal view on the metrics shown.
# Confusion Matrix:

# Predicted Negative | Predicted Positive
# Actual Negative [[560  10]
# Actual Positive [ 76  21]]

# Although this model which has been optimized for its accuracy, does contain the highest level of accuracy
# possible for this model at a level of 87.11%. Paired with a high Specificity of 98.25% which indicates that the 
# model both does its best job at how well it is able to correctly predict whether a customer is likely to churn or not,
# and its ability to correctly identify customers who are not likely to churn (Specificity).
# However the Recall/Sensitivity being at 21.65% indicates that the model does a very poor job at correctly classifying
# customers who are actually likely to churn. This paired with the confusion matrix shown, indicates to me that the inbalance
# within the classes plays a huge role, it seems to me that the model did not have enough data to truly get a feel for the
# underlying relationships. The precision score of 67.74% indicates that the model correctly identified 
# about 68% of the customers who are likely to churn, while incorrectly identifying the remaining 32% as potential churners. 
# The model may require a larger dataset to improve the Recall, feature engineering, resampling, hyperparameter tuning, or another classification model.

# end