# KNN Machine Learning example in Python

# In this example we wll be predicting customer churn from the "telecom_churn_clean.csv" dataframe.
# However the actual prediction is not what we are looking at, this is a demostration on how to compute accuracy in practice.
# Demostrating the concepts of subsetting dataframes, spliting dataframes for train/validation sets, and evaluating preformance on the accuracy metric.
# This also provides a general walkthrough of how to set up a KNN in scikit-learn and what we are doing at each step along the way.

# KNN is a Kappa Nearest Neighbors algorithm, kappa represents k or n-th number.
# KNN catagorizes an unknown instance to the average of the NN, ie. takes the majority vote of local neighbors.
# Example of prediction in a 3-NN: unknown=u, 1=yes, 2=no, 3=yes, then u=yes.
# Larger K = less complex model (takes the mean of more "votes" in the dataset) = can cause underfitting
# Smaller K = more complex model (takes mean of only one, or a few "votes") = can cause overfitting

# Imports
# pip3 install scikit-learn
# pip3 install pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

# Instantiate the model at 5-NN
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Extracting the score of our model
# The method ".score()" returns the mean accuracy. and under the hood it calls the ".predict()" method internally.
# This is why you do not see me create an instance of y_pred yet.
# Although using ".score()" is convenient, don't be confused by this notation.
# Internally it does call "y_pred = knn.predict(X_test)".
training_accuracy_score = knn.score(X_train, y_train)
testing_accuracy_score = knn.score(X_test, y_test)

# Here we want to see the metric called accuracy on our training data.
# This is important to check for overfitting on training data before we continue to the testing.
# Although it might not make sense for some to see accuracy on the training data first.
# This provides the first starting point to let us know if we need to change hyperparameters around.
# However, in this demostration I print both because I have prior experience with the model and DF and did not have to make changes.
# If you did have too high of an accuracy on the training data, you might want to change hyperparameters around or stratify, there are many solutions.
# Note accuracy = (number of correct predictions) / (total number of predictions)

# Note that the method ".format()" takes variable parameters and passes them in order to the {}'s in a string.
# The notation "{:.2%}" can be broken down as: {} tells .format() where to place variable 1.
# The colon ":" indicates the start of the format specifier.
# The period "." indicates that we want to specify a precision for the float.
# The number "2" after the period indicates that we want to round the float to 2 decimal places.
# The "%" symbol at the end of the format specifier indicates that we want to format the float as a percentage.
print("Training Accuracy of 5-NN: {} | {:.2%}".format(training_accuracy_score, training_accuracy_score))

# Printing the accuracy of the testing/validation set.
print("Testing Accuracy of 5-NN:  {} | {:.2%}".format(testing_accuracy_score, testing_accuracy_score))

# Terminal outputs

#Training Accuracy of 5-NN: 0.8960990247561891 | 89.61%
#Testing Accuracy of 5-NN:  0.8680659670164917 | 86.81%
