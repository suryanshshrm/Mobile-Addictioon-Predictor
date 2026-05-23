import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Load data
data = pd.read_csv(r"C:\Users\jatin\.vscode\mobile_addiction_dataset.csv")

#input&ouput
x = data[['ScreenTime','SocialMedia','NightUse','Unlocks']]
Y = data['Addiction']


#split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

#MOdel
model = DecisionTreeClassifier()
model.fit(X_train,Y_train)

#Prediction on test data for accuracy
y_pred = model.predict(X_test)

#Calculating Accuracy
print("Accuracy:",accuracy_score(Y_test, y_pred))

#Prediction
prediction = model.predict([[9,6,1,120]])
print(prediction)
