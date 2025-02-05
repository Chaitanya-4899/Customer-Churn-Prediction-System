# Import necessary libraries
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import pickle

#data = pd.read_csv('D:/CI Data Science Projects 2024/Customer Churn Prediction/Dataset/Telco_customer_churn.csv')

base_dir = "D:/CI Data Science Projects 2024/Customer Churn Prediction/Dataset/"
csv_dir = "D:/CI Data Science Projects 2024/Customer Churn Prediction/created CSVs/"
model_dir = "D:/CI Data Science Projects 2024/Customer Churn Prediction/saved models/"

#read all the available files / tables and create a set of all the unique columns available
list_of_files = os.listdir(base_dir)

set_of_columns_available = set()

for file in list_of_files:
    if ".csv" in file:
        df = pd.read_csv(base_dir + file)
        cols_in_df = df.columns.tolist()
        
        set_of_columns_available.update(cols_in_df)
        print("Columns in file:", file, "are", cols_in_df)
        print()
        
print("Total number of attributes / columns available :", len(set_of_columns_available))
print(set_of_columns_available)

#We can combine multiple files using Customer ID as primary key
#first read all the tables dataframes

df = pd.read_csv(base_dir + "Telco_customer_churn.csv")

#There are two ways "Customer ID" is written in column names: one with and one without space 
#fix column name to "Customer ID" in "Telco_customer_churn.csv" file
df = df.rename(columns = {'CustomerID':'Customer ID'})

list_of_csvs = ['CustomerChurn.csv', 
                'Telco_customer_churn_demographics.csv',
                'Telco_customer_churn_location.csv',
                'Telco_customer_churn_population.csv',
                'Telco_customer_churn_services.csv',
                'Telco_customer_churn_status.csv']

for file in list_of_csvs:
    temp = pd.read_csv(base_dir + file)

    if "Customer ID" in temp.columns.tolist():
        df = pd.merge(df, temp, on = "Customer ID", how = "left", suffixes=('', '_remove'))
        #df.join(temp.set_index("Customer ID"), on = "Customer ID") 
    else:
        df = pd.merge(df, temp, on = "Zip Code", how = "left", suffixes=('', '_remove'))
            
# remove the duplicate columns
df.drop([i for i in df.columns if 'remove' in i], axis = 1, inplace = True)

print("Total Number of columns : ", len(df.columns))
print("List of columns :", df.columns.tolist())
df.head()

#save the complete dataframe with all the attributes combined into a single table
df.to_csv(csv_dir + "Telecom_Customer_Churn_Complete.csv")

# replacing na values in "Churn Category" with "Not Applicable"
df["Churn Category"].fillna("Not Applicable", inplace = True)
# replacing na values in "Churn Reason" with "Not Churned"
df["Churn Reason"].fillna("Not Churned", inplace = True)

#data type of total charges column is object - as it contains null values as blank space strings
#we need to replace that with 0.0
df["Total Charges"] = np.where(df["Total Charges"] == " ", '0.0', df["Total Charges"])
df["Total Charges"] = df["Total Charges"].astype(float)

#columns in which we want to replace "No internet service" with "No"
cols_to_change = ["Online Security", "Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"]
# Applying the condition
df[cols_to_change] = np.where(df[cols_to_change] == "No internet service", "No", df[cols_to_change])

#for "Multiple Lines" column
df["Multiple Lines"] = np.where(df["Multiple Lines"] == "No phone service", "No", df["Multiple Lines"])

#group tenure in bins:
df["Tenure Bins"] = pd.cut(df['Tenure in Months'], [0, 12, 24, 48, 60, 72])
df.value_counts("Tenure Bins")

#convert numbers as strings into integers 
df["Population"] = df['Population'].str.replace(',','').astype(int)

#which so not have a clear interpretation and cannot help us determine if a customer will churn or not!!
df.set_index("Customer ID", inplace = True)

#list of columns to be droped / removed - because there is no information to be extracted from those columns / attributes
list_of_columns_to_drop = ["Count","Country","State","Churn Label","Location ID","ID","Service ID","Quarter","Status ID"
                                ,"LoyaltyID","Tenure","Tenure Months","Churn","Internet Type","Monthly Charge","Tenure in Months"
                                ,"Lat Long","Zip Code","City","Churn Reason","Churn Category","Customer Status","CLTV","Churn Score"]

list_of_columns_to_drop

df.drop(list_of_columns_to_drop, axis = 1, inplace = True)

#Add a new column to represent Satisfaction score as categorical variable for EDA 
df['Satisfaction Score Label'] = df['Satisfaction Score'].astype('category')
#rename the target column "Churn Value" to just "Churn"
df.rename(columns = {'Churn Value' : 'Churn'}, inplace = True)

df.to_csv(csv_dir + "Selected_columns_customer_churn.csv")

# Check the correlation matrix of all features
columns_for_corr = ["Monthly Charges", "Total Charges", "Population", "Avg Monthly Long Distance Charges", "Total Refunds", 
                    "Total Extra Data Charges", "Total Long Distance Charges", "Total Revenue"]
df_corr = df[columns_for_corr].corr()
sns.heatmap(df_corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


#Columns to be droped 
columns_to_be_dropped = ["Total Charges", "Monthly Charges", "Total Long Distance Charges"]
df.drop(columns_to_be_dropped, axis = 1, inplace = True)
df.drop(["Satisfaction Score Label"], axis = 1, inplace = True)


columns_to_be_encoded = []

for col in df.columns.tolist():
    if(df[col].dtype == 'object' and "Yes" in df[col].unique()):
        columns_to_be_encoded.append(col)

df[columns_to_be_encoded] = np.where(df[columns_to_be_encoded] == "Yes", 1, 0)
df[columns_to_be_encoded] = df[columns_to_be_encoded].astype(int)

df["Gender"] = np.where(df["Gender"] == "Female", 1, 0)
df["Gender"] = df["Gender"].astype(int)

df = pd.get_dummies(df, columns = ["Tenure Bins", "Offer", "Payment Method", "Contract", "Internet Service"])

# Create features and target variable
X = df.drop(["Churn"], axis = 1)
y = df["Churn"]

# using the train test split function
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 42, test_size = 0.2, shuffle = True)

#Confirm the splitting is correct:
print("Shape of Training Data : ", "\nfeatures:", X_train.shape, ", target:", y_train.shape)
print("Target Label Distribution in train set : ", "\nChurn:", y_train.sum(), "Not Churn", len(y_train) - y_train.sum())
print("Percentage of Churn:", y_train.sum() / len(y_train) * 100)
print()
print("Shape of Test Data : ", "\nfeatures:", X_test.shape, ", target:", y_test.shape)
print("Target Label Distribution in test set : ", "\nChurn:", y_test.sum(), "Not Churn", len(y_test) - y_test.sum())
print("Percentage of Churn:", y_test.sum() / len(y_test) * 100)

#format the features names:

X.index.names = ['Customer_ID']
X_train.index.names = ['Customer_ID']
X_test.index.names = ['Customer_ID']

X.columns = [col.replace(' ', '_') for col in X.columns.tolist()]
X.columns = [col.replace('(', '_') for col in X.columns.tolist()]
X.columns = [col.replace(')', '') for col in X.columns.tolist()]
X.columns = [col.replace(']', '_') for col in X.columns.tolist()]
X.columns = [col.replace(',', '') for col in X.columns.tolist()]

X_train.columns = X.columns
X_test.columns = X.columns

list_of_models = {
    'logistic_regression' : LogisticRegression(random_state = 42, max_iter = 10000),
    'Random_forest' : RandomForestClassifier(n_estimators = 150, max_depth = 4, random_state = 42),
    'XGBoost' : xgb.XGBClassifier(n_estimators = 200, max_depth = 5, random_state = 42)
}

f1_train_scores = [] 
f1_test_scores = [] 
recall_test_scores = []

#model_names = list_of_models.keys()
model_names = ['logistic_regression', 'Random_forest', 'XGBoost']


for model in model_names:
    print("\nFor Model:", model)
    
    list_of_models[model].fit(X_train, y_train)

    print("\nFor Training Set:")

    y_train_pred = list_of_models[model].predict(X_train)

    f1_train = f1_score(y_train, y_train_pred, average='macro')
    print("\nMacro F1 Score:", f1_train)

    print("\nConfusion Matrix:") 
    confusion_matrix = metrics.confusion_matrix(y_train, y_train_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.show()

    print("For Test Set:")

    y_test_pred = list_of_models[model].predict(X_test)
    
    f1_test = f1_score(y_test, y_test_pred, average='macro')
    print("\nMacro F1 Score:", f1_test)

    recall_test_score = recall_score(y_test, y_test_pred, average='macro')
    
    print("\nConfusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(y_test, y_test_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.show()
    
    f1_train_scores.append(f1_train)
    f1_test_scores.append(f1_test)
    recall_test_scores.append(recall_test_score)


pickle.dump(list_of_models['logistic_regression'], open(model_dir + 'churn_logistic_regression_model.pkl','wb'))
pickle.dump(list_of_models['Random_forest'], open(model_dir + 'churn_Random_forest_model.pkl','wb'))
pickle.dump(list_of_models['XGBoost'], open(model_dir + 'churn_XGBoost_model.pkl','wb'))


# CREATING SIMPLER MODELS FOR DEPLOYMENT
#select the columns
cols_to_drop = ['Gender', 'Partner', 'Total_Extra_Data_Charges', 'Total_Refunds', 
                'Premium_Tech_Support', 'Streaming_Music', 'Unlimited_Data',
                'Number_of_Referrals', 'Avg_Monthly_Long_Distance_Charges', 
                'Avg_Monthly_GB_Download', 'Device_Protection_Plan',
                'Number_of_Dependents', 'Streaming_TV', 'Streaming_Movies',
                'Online_Backup', 'Device_Protection', 'Dependents', 'Phone_Service', 'Multiple_Lines']
X.drop(cols_to_drop, axis = 1, inplace = True)

X_train.drop(cols_to_drop, axis = 1, inplace = True)
X_test.drop(cols_to_drop, axis = 1, inplace = True)
X_train = X_train.values
X_test = X_test.values

#Create Models 
#logistic regression Model for deployement with selected features
log_reg_model = LogisticRegression(random_state = 42, max_iter = 10000)

#fit model
log_reg_model.fit(X_train, y_train)

#make predictions on train and test set
print("\nFor Training Set:")

y_train_pred = log_reg_model.predict(X_train)

f1_train = f1_score(y_train, y_train_pred, average='macro')
print("\nMacro F1 Score:", f1_train)

print("\nConfusion Matrix:") 
confusion_matrix = metrics.confusion_matrix(y_train, y_train_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

print("For Test Set:")

y_test_pred = log_reg_model.predict(X_test)

f1_test = f1_score(y_test, y_test_pred, average='macro')
print("\nMacro F1 Score:", f1_test)

recall_test_score = recall_score(y_test, y_test_pred, average='macro')
print("\nMacro Recall Score:", recall_test_score)

print("\nConfusion Matrix:")
confusion_matrix = metrics.confusion_matrix(y_test, y_test_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

xgb_model = xgb.XGBClassifier(n_estimators = 200, max_depth = 5, random_state = 42)

#fit model
xgb_model.fit(X_train, y_train)

#make predictions on train and test set
print("\nFor Training Set:")

y_train_pred = xgb_model.predict(X_train)

f1_train = f1_score(y_train, y_train_pred, average='macro')
print("\nMacro F1 Score:", f1_train)

print("\nConfusion Matrix:") 
confusion_matrix = metrics.confusion_matrix(y_train, y_train_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

print("For Test Set:")

y_test_pred = xgb_model.predict(X_test)

f1_test = f1_score(y_test, y_test_pred, average='macro')
print("\nMacro F1 Score:", f1_test)

recall_test_score = recall_score(y_test, y_test_pred, average='macro')
print("\nMacro Recall Score:", recall_test_score)

print("\nConfusion Matrix:")
confusion_matrix = metrics.confusion_matrix(y_test, y_test_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

#Save and export models to be used in deployment 
pickle.dump(log_reg_model, open(model_dir + 'churn_logistic_regression_model_for_deployment.pkl','wb'))
pickle.dump(xgb_model, open(model_dir + 'churn_xgb_model_for_deployment.pkl','wb'))

#MODEL ENSEMBLE
y_pred_proba_log = [x[1] for x in log_reg_model.predict_proba(X_train)]
y_pred_proba_xgb = [x[1] for x in xgb_model.predict_proba(X_train)]
y_pred_proba_log = np.array(y_pred_proba_log)
y_pred_proba_xgb = np.array(y_pred_proba_xgb)

res = np.array(y_pred_proba_log + y_pred_proba_xgb) / 2.0

res = np.where(res > 0.4, 1, 0)

#make predictions on train and test set
print("\nFor Training Set:")

f1_train = f1_score(y_train, res, average='macro')
print("\nMacro F1 Score:", f1_train)

recall_train_score = recall_score(y_train, res, average='macro')
print("\nMacro Recall Score:", recall_train_score)

print("\nConfusion Matrix:") 
confusion_matrix = metrics.confusion_matrix(y_train, res)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

y_pred_proba_log = [x[1] for x in log_reg_model.predict_proba(X_test)]
y_pred_proba_xgb = [x[1] for x in xgb_model.predict_proba(X_test)]
y_pred_proba_log = np.array(y_pred_proba_log)
y_pred_proba_xgb = np.array(y_pred_proba_xgb)

y_pred_proba = (y_pred_proba_log + y_pred_proba_xgb) / 2.0

y_test_pred = np.where(y_pred_proba > 0.4, 1, 0)

print("For Test Set:")

f1_test = f1_score(y_test, y_test_pred, average='macro')
print("\nMacro F1 Score:", f1_test)

recall_test_score = recall_score(y_test, y_test_pred, average='macro')
print("\nMacro Recall Score:", recall_test_score)

print("\nConfusion Matrix:")
confusion_matrix = metrics.confusion_matrix(y_test, y_test_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

#FINAL MODEL
#fit logistic regression model
log_reg_model.fit(X, y)

#fit xgb model
xgb_model.fit(X, y)

#predict using ensemble of both the models
y_pred_proba_log = [x[1] for x in log_reg_model.predict_proba(X)]
y_pred_proba_xgb = [x[1] for x in xgb_model.predict_proba(X)]
y_pred_proba_log = np.array(y_pred_proba_log)
y_pred_proba_xgb = np.array(y_pred_proba_xgb)

y_pred_proba = (y_pred_proba_log + y_pred_proba_xgb) / 2.0

y_pred = np.where(y_pred_proba > 0.4, 1, 0)

f1 = f1_score(y, y_pred, average = 'macro')
print("\nMacro F1 Score:", f1)

recall = recall_score(y, y_pred, average = 'macro')
print("\nMacro Recall Score:", recall)

print("\nConfusion Matrix:")
confusion_matrix = metrics.confusion_matrix(y, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

#Save and export models to be used in deployment 
pickle.dump(log_reg_model, open(model_dir + 'churn_logistic_regression_model_for_deployment.pkl','wb'))
pickle.dump(xgb_model, open(model_dir + 'churn_xgb_model_for_deployment.pkl','wb'))

population_df = pd.read_csv(base_dir + 'Telco_customer_churn_population.csv', usecols = ["Zip Code", "Population"])
location_df = pd.read_csv(base_dir + 'Telco_customer_churn_location.csv', usecols = ["Zip Code", "Latitude", 'Longitude'])
population_df["Population"] = population_df['Population'].str.replace(',','').astype(int)

#merge dataframes
zip_code_map_df = pd.merge(location_df, population_df, on = "Zip Code", how = "left", suffixes=('', '_remove'))
zip_code_map_df = zip_code_map_df.drop_duplicates()
zip_code_map_df = zip_code_map_df.set_index("Zip Code")

#save the new map in a csv to be used later 
zip_code_map_df.to_csv(csv_dir + "zip_code_map_df.csv")

#print(X.columns)