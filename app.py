#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
import pickle
import numpy as np
import student
model = pickle.load(open('my_model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods = ['POST'])
def home():
    mid1 = int(request.form['m1'])
    temp1 = mid1
    mid1_total = int(request.form['m1t'])
    mid1 = int((mid1/mid1_total)*20)
    mid2 = int(request.form['m2'])
    temp2 = mid2
    mid2_total = int(request.form['m2t'])
    mid2 = int((mid2/mid2_total)*20)
    final_total = int(request.form['ft'])
    attd = int(request.form['att'])
    attd = 100 - attd
    print('mid1:', mid1)
    print('mid2:', mid2)
    arr = np.array([mid1, mid2, attd]).reshape(1,3)
    print(arr)
    pred = model.predict(arr)
    pred = np.clip(pred, 0, 20)
    pred = (pred/20) * final_total
    print('pred:', pred)
    percentage = 0
    percentage = (temp1 + temp2 + pred) / (mid1_total + mid2_total + final_total) * 100
    pgrade = ' '
    if percentage >= 90:
        pgrade = 'A+'
    elif percentage >= 86:
        pgrade = 'A'
    elif percentage >= 82:
        pgrade = 'A-'
    elif percentage >= 78:
        pgrade = 'B+'
    elif percentage >= 74:
        pgrade = 'B'
    elif percentage >= 70:
        pgrade = 'B-'
    elif percentage >= 66:
        pgrade = 'C+'
    elif percentage >= 62:
        pgrade = 'C'
    elif percentage >= 58:
        pgrade = 'C-'
    elif percentage >= 54:
        pgrade = 'D+'
    elif percentage >= 50:
        pgrade = 'D'
    else:
        pgrade = 'F'
    print(percentage)
    print(pgrade)
    predicted_final = int(round(pred[0]))
    return render_template('home.html', prediction=predicted_final, grade=pgrade)

if __name__ == "__main__":
    app.run(debug=True)