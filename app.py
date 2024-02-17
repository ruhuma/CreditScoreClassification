from flask import Flask, render_template, request, send_file
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import os
import zipfile


app = Flask(__name__)
@app.route('/')
def Hello():
    return render_template('index.html')

# Assuming your model file is in the same directory as app.py
zip_model_path = '/workspaces/CreditScoreClassification/best_rf_model.zip'
#extracted_model_path = '/workspaces/CreditScoreClassification/best_rf_model_2.pkl'
extracted_model_directory = '/workspaces/CreditScoreClassification/'
model_filename = 'best_rf_model_2.pkl'
model_path = os.path.join(extracted_model_directory, model_filename)
# Extract the model file from the ZIP archive
with zipfile.ZipFile(zip_model_path, 'r') as zip_ref:
    zip_ref.extractall(model_path)

# Your Flask route to serve the model
@app.route('/get_model')
def get_model():
    # Use the extracted model path in your application logic
    # ...

    # Serve the extracted model file
    return send_file(model_path, as_attachment=True, mimetype='application/octet-stream')

@app.route('/classification', methods = ['GET','POST'])
def hey():
    
        if request.method == 'POST':
            print('1 starting')
            selected_month = float(request.form['month'])
            print('month',selected_month)
            age = float(request.form['age'])
            print('age',age)
            occupation = request.form['occupation']
            annualincome = float(request.form['annualincome'])
            print('annualincome',annualincome)
            num_bank_accounts = float(request.form['num_bank_accounts'])
            num_credit_card = float(request.form['num_credit_card'])
            interest_rate = float(request.form['interest_rate'])
            num_loans = float(request.form['num_loans'])
            print('2')
            delay_due_date = float(request.form['delay_due_date'])
            delay_payments = float(request.form['delay_payments'])
            changed_credit_limit = float(request.form['changed_credit_limit'])
            credit_inquiry = float(request.form['credit_inquiry'])
            credit_mix = float(request.form['credit_mix'])
            outstanding_debts = float(request.form['outstanding_debts'])
            credit_ration = float(request.form['credit_ration'])
            credit_history_age = float(request.form['credit_history_age'])
            minimum_amount = float(request.form['minimum_amount'])
            EMI_per_month = float(request.form['EMI_per_month'])
            amount_invested_monthly = float(request.form['amount_invested_monthly'])
            monthly_balance = float(request.form['monthly_balance'])
            payment_behaviour = float(request.form['payment_behaviour'])
            print(payment_behaviour)
            print(occupation)
            with open('label_encoder_2.pkl', 'rb') as file:
                label_encoder = pickle.load(file)
            occupation_encoded = label_encoder.transform(np.array([[occupation]]))
            print('encoded',occupation_encoded)
            occupation_encoded_d = occupation_encoded[0].astype(np.float64)
            print(occupation_encoded_d)
            print(monthly_balance)

            input_array = np.array([[selected_month,age,occupation_encoded_d,annualincome,num_bank_accounts,num_credit_card,interest_rate
                                     ,num_loans,delay_due_date,delay_payments,changed_credit_limit,credit_inquiry,credit_mix,outstanding_debts,credit_ration,credit_history_age,minimum_amount,
                                     EMI_per_month,amount_invested_monthly,payment_behaviour,monthly_balance]]).astype(np.float64)
            
            scaler = joblib.load("stdscaler_CS.pkl")
            input_scaled_G = scaler.transform(input_array)
            print('Here is again after scaling')
            #model = pickle.load(open(extracted_model_path,'rb'))
            print('model path',model_path)
            path_model = model_path+'/best_rf_model.pkl'
            print('model path',path_model)
            #with open(path_model, 'rb') as file:
             #   model = pickle.load(file)
            model = pickle.load(open(path_model,'rb'))
            #('file=',file)
            print('Input scaled',input_scaled_G)
            creditscoreclassified = model.predict(input_scaled_G)
            print('Input Array',input)
            print('Scaled Array',input_scaled_G)
            print(creditscoreclassified)
            if creditscoreclassified == 0:
                classification = 'Good'
            elif creditscoreclassified== 1:
                classification = 'Bad'
            elif creditscoreclassified== 2:
                classification = 'Standard'
            else:
                classification = 'Unable to process'
            return render_template('classification.html',creditscore=classification)
        else:
            return render_template('error.html',creditscore='Unable to process with received input.')
    #except ValueError:
     #   return render_template('error.html',creditscore='Invalid Input! Please enter correct data types in the form!')

if __name__ == '__main__':
    app.run()
