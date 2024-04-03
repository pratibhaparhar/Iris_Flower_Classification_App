from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score
import pickle

data=load_iris()
# print(data)

x,y=load_iris(return_X_y=True)
# print(x)
# print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.70,random_state=42)

# model load
log_reg=LogisticRegression()

# model train

log_reg.fit(x_train,y_train)

# prediction
# log_reg.predict(x_test)

x1_sepal_length = float(input('x1_sepal_length:'))
x2_sepal_width = float(input('x2_sepal_width:'))
x3_petal_length =float(input('x3_petal_length:'))
x4_petal_width =  float(input('x4_petal_width:'))

output = log_reg.predict([[x1_sepal_length,x2_sepal_width, x3_petal_length, x4_petal_width]])
print(f"Predictipn = {data['target_names'][output]}" )

pickle.dump(log_reg,open('log_reg_model.pkl', 'wb'))