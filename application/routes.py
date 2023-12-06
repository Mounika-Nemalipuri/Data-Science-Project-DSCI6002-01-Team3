import requests
from application import app
from flask import render_template, request, json, jsonify,Response, url_for
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, RobustScaler 
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,mean_squared_error, r2_score




# import requests
from fastapi import FastAPI, HTTPException
import numpy as numpy
req_app = FastAPI()

import pandas as pd


#decorator to access the app
@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

#decorator to access the service
@app.route("/carclassify", methods=['POST'])
def carclassify():
    try:
        #extract form inputs
        buying = request.form.get("buying")
        maint = request.form.get("maint")
        doors = request.form.get("doors") 

        buy1 = request.form.get("buy1")
        buy2 = request.form.get("buy2")
        buy3 = request.form.get("buy3")
        buy4 = request.form.get("buy4")
        buy5 = request.form.get("buy5")

        input_data = json.dumps({"buying": buying, "maint": maint, "doors": doors,"buy1": buy1 ,"buy2":buy2,"buy3":buy3,"buy4":buy4,"buy5":buy5})

        # Input validation could be added here
        # url = url_for('api') 
        # Perform prediction
        
        results =  requests.post('http://127.0.0.1:5000/apii', input_data)
        c=results.content.decode('UTF-8')
        c=json.loads(c)
        #send input values and prediction result to index.html for display
        return render_template("index.html", buying=buying, maint=maint, doors=doors,
                               results=c,buy1=buy1,buy2=buy2,buy3=buy3,buy4=buy4,buy5=buy5)
    except Exception as e:
        # Log the error and handle it gracefully
        print(f"An error occurred: {e}")
        return render_template("index.html", error_message="An error occurred. Please try again.")
    
@app.route('/apii', methods=['GET', 'POST'])
def predict(): 
    
    data = request.get_json(force=True)
    # requestData = numpy.array([ data["buying"], data["maint"],data["doors"]]) 
    requ = numpy.array([data["buying"]]).astype(float)[0]
    re1 = numpy.array([data["maint"]]).astype(int)[0]
    re3 = numpy.array([data["doors"]]).astype(float)[0]

    de1 = numpy.array([data["buy1"]]).astype(float)[0]
    de2 = numpy.array([data["buy2"]]).astype(float)[0]
    de3 = numpy.array([data["buy3"]]).astype(float)[0]
    de4 = numpy.array([data["buy4"]]).astype(float)[0]
    de5 = numpy.array([data["buy5"]]).astype(float)[0]

     
    # print(requ,re1,re3)
    
    z = pd.read_csv(r'Clustered_data/customer_data.csv')
    X = z[['count', 'review_score', 'price']]
    y = z[['Discount', 'cluster']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    base_regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    multioutput_regressor = MultiOutputRegressor(base_regressor)
    multioutput_regressor.fit(X_train, y_train)
    prediction = multioutput_regressor.predict([[requ,re1,re3]])


    mse = mean_squared_error(y_test, multioutput_regressor.predict(X_test))
    # print(balance_data)
    if (prediction[0][1] == 0):
        c = "Need Advertisement - As the review is 'Satisfied', price is 'Average' and count is 'Less' . The product need further advertisement to increase the sale."
    elif(prediction[0][1] == 1):
        c="Need Advertisement - As the review is 'Satisfied', price is 'Low' and count is 'Average' . The product need further advertisement to increase the sale. Later based on the sales increase, can increase the price of the product also."
    elif(prediction[0][1]==2):
        c="Quality Improve - As the review is 'Not Satisfied' & 'Not Bad' , price is 'Low' and count is 'Low' . The product needs to improvise in Quality."
    elif(prediction[0][1] == 3):
        c="Reasonable - As the review is 'Satisfied' & 'Good' , price is 'More than average' and count is 'Low' . The product is reasonable since review is good."
    elif(prediction[0][1]==4):
        c="Reasonable - As the review is 'Satisfied' & 'Good' , price is 'Less' and count is 'More than average' . The product is reasonable since review is good."
    elif(prediction[0][1]==5):
        c="Not Reasonable - As the review is 'Not Satisfied', price is 'More than average' and count is 'Less' . The product is not reasonable since review is bad and price is more than average."
    elif(prediction[0][1]==6):
        c="Need Advertisement - As the review is 'Satisfied', price is 'More than average' and count is 'Less' . The product need further advertisement to increase the sale."
    elif(prediction[0][1]==7):
        c="Reasonable - As the review is 'Not Bad' & 'Good' , price is 'Average' and count is 'More than average' . The product is reasonable since review is good."
    
    delivery_data = pd.read_csv(r'Clustered_data/delivery_processed.csv')
    
    skewed_preprocess = Pipeline([
    ('binning',KBinsDiscretizer(n_bins=30,encode='ordinal')),
    ('scaling',RobustScaler(quantile_range=(0,100)))
    ])

    tra_X = ColumnTransformer([
        ('skewed',skewed_preprocess,['freight_value','product_weight_g','product_length_cm','product_width_cm','delivery_distance']),
    ],remainder='passthrough')

    transform_val = tra_X.fit_transform(delivery_data[['freight_value','product_weight_g','product_length_cm','product_width_cm','delivery_distance']])
        
    y1 = pd.cut(delivery_data['estimation_diff'],bins=7,duplicates='drop',labels=[1,2,3,4,5,6,7])

    X1 = pd.DataFrame(data={
        'freight_value':transform_val[:,0],
        'product_weight_g':transform_val[:,1],
        'product_length_cm':transform_val[:,2],
        'product_width_cm':transform_val[:,3],
        'delivery_distance':transform_val[:,4]
            
    })


    X_train, X_test, y_train, y_test = train_test_split(X1, y1,test_size=0.2, random_state=42)

    model = LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter=1000)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    y_pred1 = model.predict(X_train)


    accuracy_Testing = accuracy_score(y_test, y_pred)
    accuracy_Trainging = accuracy_score(y_train, y_pred1)
    confusion = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred,zero_division=1)
    predic2 = model.predict([[de1,de2,de3,de4,de5]])
    print('ppp',predic2)

    if predic2[0] == 1 :
        c2= "5-3 Months before estimation"
    if predic2[0] == 2 :
        c2 = "3-1 Months Before estimation"
    if predic2[0] == 3 :
        c2 = "1 Months -1 week Before estimation"
    if predic2[0] == 4 :
        c2 = "1Week - 1 Month delayed after estimation"
    if predic2[0] == 5 :
        c2 = "1 -3 Months delayed after estimation"
    if predic2[0] == 6 :
        c2 = "3 - 5 Months delayed after estimation"
    if predic2[0] == 7 :
        c2 = "5-6 Months Delayed after estimation"



    return Response(json.dumps([prediction[0][0],c,mse,c2,accuracy_Testing]))


