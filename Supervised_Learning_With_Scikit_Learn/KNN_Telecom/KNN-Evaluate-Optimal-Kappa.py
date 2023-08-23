# KNN Machine Learning example in Python

# In this example we wll be predicting customer churn from the "telecom_churn_clean.csv" dataframe.
# This is a demostration of subsetting dataframes, spliting dataframes for train/validation sets
# , evaluating preformance at scale for the accuracy metric to determine optimal Kappa value, and visualization.

# KNN is a Kappa Nearest Neighbors algorithm, kappa represents k or n-th number.
# KNN catagorizes an unknown to the avg of the NN, ie. takes the majority vote of local neighbors.
# Example of prediction in a 3-NN: unknown=u, 1=yes, 2=no, 3=yes, then u=yes.
# Larger K = less complex model (takes the mean of more "votes" in the dataset) = can cause underfitting
# Smaller K = more complex model (takes mean of only one, or a few "votes") = can cause overfitting

# Imports
# pip3 install scikit-learn
# pip3 install pandas
# pip3 install matplotlib
# pip3 install numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

# ----------Locating Optimal Kappa----------

# Creating a range so that we can find the most optimal K value to use for the KNN model.
# np.arange() takes parameters (start, stop, step) and returns an ndarray.
# This produces: [1, 2, 3, 4, 5, ..., 20]
neighbors = np.arange(1, 21)

# Initializing empty dictionaries to store accuracy values
train_accuracies = {}
test_accuracies = {}

# Loop through the potential K values and load them into the dictionaries
for neighbor in neighbors:
  
	# Set up a KNN Classifier.
	# We do this within the loop because we are changing the K classification each iteration.
	knn = KNeighborsClassifier(n_neighbors= neighbor)
  
	# Fit the model
	# This is also in the loop because knn is reformatted each iteration.
	knn.fit(X_train, y_train)
  
	# Compute accuracy and places it in the respective location in the dictionaries
	# Adds the key-value pair to the dictionary
	train_accuracies[neighbor] = knn.score(X_train, y_train)
	test_accuracies[neighbor] = knn.score(X_test, y_test)

# Display the raw output of the work we just did.
print("neighbors: {} \n train_accuracies: {} \n test_accuracies: {}".format(neighbors, train_accuracies, test_accuracies))

# ----------Visualization----------

# Now we will start creating our graph for visualization.
# The numbers here just deal with the dimensions of the graph I wanted to make.
plt.figure(figsize=(12, 8))

# Add a title
plt.title("KNN: Varying Number of Neighbors")

# Plot training accuracies
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")

# Plot test accuracies
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")

# Queries the dictionary for the highest accuracy value and the key that pairs with it by using the ".get" attribute method.
max_value = max(test_accuracies.values())
max_kappa = max(test_accuracies, key=test_accuracies.get)

# Displays a horizonal line showing the intersection of the most optimal Kappa value to use.
plt.axhline(max_value, color='r', linestyle='--')

# Shows information on the optimal Kappa line
plt.text(max_kappa, max_value, f'Optimal Kappa: {max_kappa} Accuracy: {max_value:.2%}', ha='center', va='bottom')

# Plot legend box, along with the x and y lables
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

# Display the plot
plt.show()

# Print to the terminal what our results are
print("Optimal Kappa: {} \nAccuracy: {:.2%}".format(max_kappa, max_value))

# Terminal Output

# Optimal Kappa: 11 
# Accuracy: 87.11%