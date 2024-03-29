import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

data_dict = pickle.load(open('data.pickle','rb'))

x = np.asarray(data_dict['data'])
y = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,shuffle=True,stratify=y)

model = RandomForestClassifier() #it is'nt working because random forest can only classify 2 dimensional things and our image is 3 dimensional

pipe = Pipeline([('Scaler',StandardScaler()),('svc',SVC(kernel='rbf',C=10))])

model.fit(x_train,y_train)

y_predict = model.predict(x_test)

print(y_predict)
print(y_test)

print(accuracy_score(y_predict,y_test))

f = open('model.pickle', 'wb')
pickle.dump({'model': model}, f)
f.close()