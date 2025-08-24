import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

csv = pd.read_csv('wine_quality_merged.csv', sep=',',encoding='utf-8')

pipeline = Pipeline([('scaler',StandardScaler()),
                     ('svm',SVC(kernel='linear', C=1, random_state=42)
                      )])

csv.loc[csv["type"] == 'white', "type"] = 1
csv.loc[csv["type"] == 'red', "type"] = 0

csv['type'] = csv['type'].astype('int')

X = csv[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality']]

y = csv['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

score = cross_val_score(pipeline, X_train, y_train, cv=5)
print(score)
score_final = cross_val_score(pipeline, X_test, y_test, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (score.mean(), score.std()))
print(score_final)
print("%0.2f accuracy with a standard deviation of %0.2f" % (score_final.mean(), score_final.std()))