import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
#data Clean
data=pd.read_csv("E:/Abdelfattah/DATASET/train.csv")
data=data.drop(["Cabin","PassengerId","Ticket","Name"],axis=1)
data['Age']=data['Age'].fillna(data['Age'].mean())
#edit Data
data_object=data.select_dtypes(include=["object"])
data_unobject=data.select_dtypes(exclude=["object"])
#encoder
la=LabelEncoder()
for i in range (data_object.shape[1]):
  data_object.iloc[:,i]=la.fit_transform(data_object.iloc[:,i])
  full_data=pd.concat([data_object,data_unobject],axis=1)
#edit
x=full_data.iloc[:,-7:]
y=data.iloc[:,:-7]
#test&train
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.95)
##classification model
model=LogisticRegression()
model.fit(x_train,y_train)
#predict

predictions = model.predict(x_test)
print("predict=",predictions)  
#confusion matrix
con=confusion_matrix(y_test,predictions)
print("Confustion matrix",con)
#classification rebort
cl=classification_report(y_test, predictions)
print("classification_report",cl)
cor=full_data.corr()
heat=sns.heatmap(cor,annot=True)

print("Train Score=",model.score(x_train,y_train)*100,"%")
print("Test Score=",model.score(x_test,y_test)*100,"%")


