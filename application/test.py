from application import app
from flask import render_template, request, json, jsonify
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import pandas as pd

#decorator to access the app
@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

#decorator to access the service
@app.route("/carclassify", methods=['GET', 'POST'])
def carclassify():

    #extract form inputs
    buying = request.form.get("buying")
    maint = request.form.get("maint")
    doors = request.form.get("doors")
    persons = request.form.get("persons")
    lug_boot = request.form.get("lug_boot")
    safety = request.form.get("safety")

    #extract data from json
    input_data = json.dumps({"buying": buying, "maint": maint, "doors": doors, "persons": persons, "lug_boot": lug_boot, "safety": safety})

    z = pd.read_csv(r'Clustered_data/customer_data.csv')
    X = z[['count','review_score','price']]
    y = z[['Discount','cluster']]

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    base_regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    multioutput_regressor = MultiOutputRegressor(base_regressor)

    multioutput_regressor.fit(X_train, y_train)
    y_pred = multioutput_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    p=multioutput_regressor.predict([[buying, maint,persons]])

    c=mse
    #send input values and prediction result to index.html for display
    return render_template("index.html", buying = buying, maint = maint, doors = doors, persons = persons, lug_boot = lug_boot, safety = safety, results=p)
  
