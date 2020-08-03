import numpy as np
from flask import Flask, request, jsonify, render_template

import joblib
import pandas as pd
from flask import Flask, request, render_template

model = joblib.load('model.pkl.z')
app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

df = None
@app.route("/")
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        global df
            
        import pandas as pd
        df = pd.DataFrame(columns =['Term', 'NoEmp','CreateJob', 'RetainedJob','GrAppv','SBA_Appv',
                           'City', 'State', 'Bank', 'BankState', 'NewExist','FranchiseCode', 'UrbanRural', 'RevLineCr','LowDoc'])
        int_col_names= ['Term', 'NoEmp','CreateJob', 'RetainedJob','GrAppv','SBA_Appv']
        for i in int_col_names:
            df.loc[0,f'{i}'] = int(request.form[f'{i}'])
            
        char_col_names= ['City', 'State', 'Bank', 'BankState', 'NewExist','FranchiseCode', 'UrbanRural', 'RevLineCr','LowDoc']
        for i in char_col_names:
            df.loc[0,f'{i}'] = request.form[f'{i}']
        return render_template('home_1.html',shape = df.shape)
      
    return render_template('home_1.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
		
         
        pre = model.predict(df)
    return render_template('re.html', prediction = pre)





if __name__ == '__main__':
    app.run(debug=True)
