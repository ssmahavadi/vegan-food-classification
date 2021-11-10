import pandas as pd # data processing
import numpy as np # Using arrays
import matplotlib.pyplot as plt # visualization
from matplotlib import rcParams # Graph size
#from termcolor import colored as cl # Text customization

from sklearn.tree import DecisionTreeClassifier as dtc # Tree algorithm
from sklearn.model_selection import train_test_split # Split data
from sklearn.metrics import accuracy_score # Model accuracy
from sklearn.tree import plot_tree # Tree chart

rcParams['figure.figsize'] = (30, 25)

df = pd.read_csv('dataset(specialcases).csv') # or 'dataset(original).csv'
df.drop('VeganFood', axis = 1, inplace = True)

print((df.head()))

df.info()

for i in df.BrownGreen.values:
    if i  == 'neither':
        df.BrownGreen.replace(i, 0, inplace = True)
    elif i == 'green':
        df.BrownGreen.replace(i, 1, inplace = True)
    elif i == 'brown':
        df.BrownGreen.replace(i, 2, inplace = True)

for i in df.Breakfast.values:
    if i == 'yes':
        df.Breakfast.replace(i, 0, inplace = True)
    elif i == 'no':
        df.Breakfast.replace(i, 1, inplace = True)

for i in df.Taste.values:
    if i == 'sweet':
        df.Taste.replace(i, 0, inplace = True)
    elif i== 'bitter':
        df.Taste.replace(i, 1, inplace = True)

for i in df.Spherelike.values:
    if i == 'yes':
        df.Spherelike.replace(i, 0, inplace = True)
    elif i == 'no':
        df.Spherelike.replace(i, 1, inplace = True)

for i in df.TreePlant.values:
    if i == 'tree':
        df.TreePlant.replace(i, 0, inplace = True)
    elif i == 'plant':
        df.TreePlant.replace(i, 1, inplace = True)

        
print(df)

X_var = df[['BrownGreen', 'Breakfast', 'Taste', 'Spherelike', 'TreePlant']].values # independent variable
y_var = df['Classification'].values # dependent variable

print(('X variable samples : {}'.format(X_var[:-1])))
print(('Y variable samples : {}'.format(y_var[:-1])))

X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.2, random_state = 0)

print(('X_train shape : {}'.format(X_train.shape)))
print(('X_test shape : {}'.format(X_test.shape)))
print(('y_train shape : {}'.format(y_train.shape)))
print(('y_test shape : {}'.format(y_test.shape)))

model = dtc(criterion = 'gini', max_depth = 3) # or entropy
model.fit(X_train, y_train)

pred_model = model.predict(X_test)

print(('Accuracy of the model is {:.0%}'.format(accuracy_score(y_test, pred_model))))

feature_names = df.columns[:-1]
target_names = df['Classification'].unique().tolist()

plot_tree(model, 
          feature_names = feature_names, 
          class_names = target_names, 
          filled = True, 
          rounded = True)
          
plt.savefig('tree2(gini).png')
