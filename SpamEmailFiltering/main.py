import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm 


spam_data = pd.read_csv('spam.csv')     # ham -> non spam emails
                                        # spam -> spam emails

# taking a dataset and dividing it into two separate datasets
# first dataset : training dataset -> to fit the machine 
# second dataset : test dataset -> to evaluate the fit of the ML model

# steps:    1) fit the model on available data
#           2) make predictions based on new examples (target values)
#           3) comparing the predictions against the actual output

# EmailText column
txt = spam_data['v2']
# Label column
label = spam_data["v1"]      # later, must be predict based on the model

txt_train, txt_test, label_train, label_test = train_test_split(txt,
                                                                label,
                                                                test_size=0.2   # sets the testing set to 20% of "txt" and "label"
                                                                )

# Extracting Features
#--------------------
cv = CountVectorizer()  # randomely assigns a number to each word = tokenizing
                        # then it counts the number of occurrences of words

features = cv.fit_transform(txt_train)  # randomly assigns a number to each word
                                        # it counts the number of occurences of each word
                                        
# output -> (email_index, word_recognization_id)  number_of_occurences  


# Building the Model
#-------------------
# creating a SVM (Support Vector Machine) algorithm:
#       a linear model for classification and regression
model = svm.SVC()
model.fit(features, label_train)    # trains the model with "features" and "label_train"

# then it checks the prediction against the "label_train" and adjusts its parameters until it reaches the highest possible accuracy



# Testing our email spam detector
#--------------------------------
features_test = cv.transform(txt_test)      # makes predictions from "txt_test"
print("Accuracy: {}".format(model.score(features_test, label_test)*100))



