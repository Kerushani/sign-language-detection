# classifier - machine learning model that learns patterns in data so that it can predict labels later
# training means to give classifier examples along with correct answers so model analyzes and learns to distinguish

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np    

data_dict = pickle.load(open("./data.pickle", "rb"))
# turns incoming data from data.pickle to numpy array - does fast numerical operations on arrays and matrices
# expected_length = 30

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

#SPLIT DATA
#refer back to "training" - here x_train and y_train uses 80% of the data so the model can learn
#x_test and y_test uses 20% of the data to let the model test -> as defined in test_size
x_train, x_test, y_train, y_test = train_test_split(data,labels, test_size=0.2, shuffle=True, stratify=labels)

#CREATE MODEL
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print("{}% of samples were classified correctly!".format(score*100))

f = open("model.p", "wb")
pickle.dump({"model": model}, f)
f.close()