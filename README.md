# Data-Science-Project-DSCI6002-01-Team3
Introduction
Olist is a player in the Brazilian e-commerce industry, which is a rapidly 
evolving field with unpredictable dynamics. In order to enhance their sales, a 
majority of companies in the global E-commerce sector are embracing a 
customer-centric strategy. Analyzing customer data to check for any existing 
gap between customer expectation and the service provided by Olist and 
understanding the reason for the gap can spur Olist’s sales.
Feedback is the cornerstone of customer-centricity. Olist collects feedback 
from its customers in the form of a review score and a review message. 
The review score is assessed on a scale ranging from one to five, where one 
signifies the least favorable customer feedback about the service, while five 
denotes the most positive customer perception. Understanding current 
patterns in the review score and finding a way to predict customer review 
score, based on data about the product and order, can springboard Olist’s 
revenue.
II. Executive Summary
Adopting a customer-centric approach is crucial for companies in the Ecommerce industry to ensure the success of their business model. Feedback 
based business decisions have seen remarkable growth in-terms of business 
reach, product sales, and optimization of supply networks. For this reason, we 
chose to select a dataset provided by Olist Stores, a prominent e-commerce 
website based in Brazil.
The goal of this project is to utilize the sales data along with the customer 
feedback to understand possible correlations and build a classification model 
to understand the trend in the data. For this, we used multiple machine 
learning classification algorithms to assess the predicted data to the actual 
data. Finally, evaluating each classification model based on their performance 
parameters and selecting the best model that would represent the data.
III. Background Theory
Olist, the leading department store in the Brazilian marketplaces, generously 
shared this dataset. Olist serves as a bridge between small businesses across 
Brazil and various channels, eliminating any complexities with a streamlined, 
single contract. Through the Olist Store, merchants can effortlessly showcase 
and sell their products, while relying on Olist's logistics partners for seamless 
delivery directly to customers.
Upon the completion of a purchase on Olist Store, the seller is promptly 
notified to proceed with fulfilling the order. Subsequently, once the customer 
has received the product or the estimated delivery date has arrived, a 
satisfaction survey is sent via email to the customer. This survey allows the 
customer to rate their purchasing experience and provide additional 
comments if desired.
IV. Data Description & Exploration 
This project analyzed and predicted e-commerce orders using a dataset of 
100,000 orders made at Olist Store, a major Brazilian e-commerce platform, 
from 2016 to 2018. The dataset provided information from multiple
perspectives:
• Orders: Status, price, payment method, freight details, customer 
location.
• Products: Category, attributes, reviews.
• Customers: Demographics, purchase history

V. Methodology
Data wrangling - Data wrangling is the essential process of transforming and 
organizing intricate and untidy datasets, making them more accessible and suitable 
for analysis. 
EDA – It’s a short for Exploratory Data Analysis, entails employing various 
techniques, predominantly graphical, to gain insights and understand patterns within 
the data. 
Modeling – It involves training a machine learning algorithm to predict labels based 
on features, optimizing it for specific business requirements, and validating it using 
holdout data. The outcome of modeling is a trained model that can be utilized for 
making predictions on new data points. 
Model Evaluation - Model evaluation occurs after splitting the data into training 
and testing sets and assessing the model's performance. Finally, deployment 
involves integrating ML/DL models into an existing production environment, enabling 
practical business decision-making by accepting input and providing output.
Deployment - Deployment of ML/DL model refer to the process of integrating the 
models into an existing production environment. This integration enables the system 
to accept input data and provide an output based on the model's predictions. In 
other words, it allows the model to be utilized in a real-world setting to make 
accurate and automated decisions. The deployment ensures that the model is 
available for use by end-users and can seamlessly interact with other components of 
the production environment to deliver the desired results

Key Findings:
• Customer behavior: Most orders originated from Sao Paulo, with credit card 
the preferred payment method. Customers spent an average of 125 Brazilian 
Real per order.
• Product trends: Certain product categories saw higher demand, while some 
received consistently negative reviews.
• Price and freight: Average product value correlated with customer 
satisfaction. High freight costs could negatively impact customer experience


Modeling:
The objective of the model is to predict the review score of the dataset 
according to the variables of the dataset and to recommend the variable that 
matters to the customer the most which will be significant to the review score 
too. Also, we implement this model to understand logically if the customers 
churn or not.
We used Logistic Regression model for prediction & K-means Clustering for 
classification of the reviews.
To run the logistic regression model, we start off by creating the training 
dataset and the validation dataset, both consisting of 80% and 20% of the 
dataset respectively. As it is a classification regression model, we need to give 
the levelling reference for it to give the logistic regression results. We run the 
logistic regression where it is a multinomial regression, and we get the 
following results:



Results:
This project successfully analyzed and predicted e-commerce customer 
preferences & behavior using K-means clustering and Logistic Regression with 
Flask deployment. Logistic regression achieved an accuracy of 89% in 
predicting customer actions (purchase or churn) based on their reviews. This 
model proved highly effective in identifying potential churn risk and 
opportunities for targeted marketing.
Overall, this information empowers businesses to personalize their marketing 
strategies and improve customer engagement. The Flask deployment provides 
a readily available tool for implementing these insights.
