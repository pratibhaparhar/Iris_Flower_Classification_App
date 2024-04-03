from flask import Flask,request,render_template
import pickle

app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template('index.html')
    # return "<center><h1>Welcome to Iris Classification App<h1><center>"


@app.route('/prediction',methods=['POST'])
def prediction():
    data=request.form

    x1_sepal_length = float(data['sepal_length'])
    x2_sepal_width = float(data['sepal_width'])
    x3_petal_length =float(data['petal_length'])
    x4_petal_width = float(data['petal_width'])

    user_input=[[x1_sepal_length,x2_sepal_width,x3_petal_length,x4_petal_width]]

    log_reg=pickle.load(open('log_reg_model.pkl','rb'))

    target = ['setosa', 'versicolor', 'virginica']

    output=log_reg.predict(user_input)

    prediction=target[output[0]]
    
    return render_template('result.html',prediction_result=prediction)

if __name__=="__main__":
    app.run(debug=True)