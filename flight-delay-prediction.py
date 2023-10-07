#!/usr/bin/env python
# coding: utf-8

# # Flight Delay Prediction
# We aim to predict if a flight will be delayed with available data from the airport and airline.

# ## 1. Data
# 
# - Kaggle dataset on [airlines delay](https://www.kaggle.com/datasets/giovamata/airlinedelaycauses?datasetId=355&sortBy=voteCount)
# 
# - From the U.S. Department of Transportation's (DOT) Bureau of Transportation Statistics (BTS) tracks the on-time performance of domestic flights operated by large air carriers

# In[ ]:


import joblib
import gdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from sklearn.base import clone
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, classification_report, 
    f1_score, plot_confusion_matrix, confusion_matrix, 
    roc_curve, PrecisionRecallDisplay, roc_auc_score,
    precision_recall_curve, RocCurveDisplay
)
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, 
    train_test_split, HalvingGridSearchCV
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")


# In[ ]:


gdrive_dataset_url = "https://drive.google.com/uc?id=11C42HscaDibTpEk4EfMmiDPEgkva01Tc"
gdown.download(gdrive_dataset_url, "archive.zip", quiet=False)


# In[ ]:


get_ipython().system('unzip /content/archive.zip')


# In[ ]:


df = pd.read_csv("/content/DelayedFlights.csv")

# 1.1 Drop first column and observe dimensions
df = df.drop(["Unnamed: 0"], axis=1)

print('Dataframe dimensions:', df.shape)

df_flights = df.copy()
df.head()


# In[ ]:


# 1.2 Create info_df to record column attributes

tab_info = pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})

tab_info = tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null value (count)'}))
tab_info = tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]).T.rename(index={0:'null values (%)'}))
tab_info = tab_info.append(pd.DataFrame(df.describe().astype(int)))

tab_info = tab_info.T

tab_info


# In[ ]:


# 1.3 Categorise variables (reference: Flight delay prediction based on deep learning and Levenberg‐Marquart algorithm)

date_list = ['Year','Month','DayofMonth','DayOfWeek']
delay_time_list = ['DepTime','DepDelay','ArrTime','ArrDelay	']
delay_reason_list = ['CarrierDelay','WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay']
flight_info_list = ['FlightNum','Origin','Dest']


# **Observations**:  
# Only when Arrival Delay is longer than 15 minutes there's data about what caused the delay.   
# Arrival Delay is the sum of CarrierDelay, WeatherDelay, NASDelay and LateAircraftDelay. In cases of cancelation or diversion there's no data related to delay causes. Thus these columns can be imputed with 0.

# ## 2. EDA

# ### 2.1 Label

# In[ ]:


# Add new date and delay variables
df['FlightDate'] = pd.to_datetime(df['Year'].astype(str) + '/' + df['Month'].astype(str) + '/' + df['DayofMonth'].astype(str))

# counted as delay if > 15 min
df['DepDel15'] = np.where(df['DepDelay'] > 15,1,0)
df['ArrDel15'] = np.where(df['ArrDelay'] > 15,1,0)


# In[ ]:


df.hist(column='DepDel15')


# Since we are predicting delays, using a threshold of 15 minutes to categorize a delay leads to the delays being the majority class. In view of this, we will experiment with another threshold for delays while at the same time predicting severe delays.

# In[ ]:


df['DepDel30'] = np.where(df['DepDelay'] > 30,1,0)
df['DepDel30'].value_counts()


# Using a threshold of 30 minutes leads to the delays being the minority class and at the same time prediting longer delays might have a deeper business application.

# In[ ]:


df.hist(column='DepDel30')


# ### 2.2 Exploring Arr and Dep Delay

# In[ ]:


plt.figure(figsize = (10,10))

plt.subplot(2,2,1)
sns.distplot(df['ArrDelay'])

plt.subplot(2,2,2)
sns.distplot(df['DepDelay'])

plt.show()


# Delays are mostly located on the left side of the graph, with a long tail to the right. The majority of delays are short, and the longer delays, while unusual, are more heavy loaded in time.

# ### 2.3 Group DepDelay by TailNum

# In[ ]:


# plane tail number = aircraft registration, unique aircraft identifier
# Hypothesis: some aircrafts are more advanced and less likely to delay

def get_stats(group):
    return {'count': group.count(), 'min': group.min(), 'max': group.max(), 'mean': group.mean(), 'std': group.std(), 'sum': group.sum()}

tailnum_stats = df['DepDelay'].groupby(df['TailNum']).apply(get_stats).unstack().sort_values('count', ascending = False)

plt.figure(figsize = (20,20))

plt.subplot(2,2,1)
plt.pie(tailnum_stats['mean'][:10], labels = tailnum_stats.index[:10])

plt.subplot(2,2,2)
plt.pie(tailnum_stats['mean'][-9:], labels = tailnum_stats.index[-9:])

tailnum_stats


# Popular aircrafts on average have a consistent delay of approximaltely 30 min while less popular aircrafts have a fluctuating delay time.

# ### 2.4 Dest groupby DepDelay

# In[ ]:


# Hypothesis: some destinations are busier & more likely to delay

dest_stats = df['DepDel15'].groupby(df['Dest']).apply(get_stats).unstack().sort_values('count', ascending = False)
dest_stats.sort_values(['count','mean'], ascending = False)


# Popular destinations are more likely to get delay perhaps due to traffic congestion

# ### 2.5 Frequency of Originating and Departing Airports
# Viewing top 20 Originating and Departing Airports

# In[ ]:


plt.figure(figsize = (20,20))

plt.subplot(2,2,1)
sns.countplot(x='Origin', data=df, order=df['Origin'].value_counts().iloc[:20].index)

plt.subplot(2,2,2)
sns.countplot(x='Dest', data=df, order=df['Dest'].value_counts().iloc[:20].index)


# A majority of flights are occuring between ATL, ORD, DFW, DEN, and LAX.

# ### 2.6 Seasonal Effect
# Month and DayOfWeek

# In[ ]:


plt.figure(figsize = (20,20))

plt.subplot(2,2,1)
sns.countplot(x = 'Month', palette='Set1', data = df)

plt.subplot(2,2,2)
sns.countplot(x = 'DayOfWeek', palette='Set2', data = df)


# Month effect:  
# There is more data for the months of January and December 2008. This can be expected as travelling is popular during the end of the year. The summer months also see an uptick in flights data.
# 
# Week effect:  
# It seems that there are most fligths on Friday (5) according to our datset.

# ### 2.7 Nulls

# In[ ]:


sns.heatmap(df.isnull(),cbar=False,cmap='plasma')


# ### 2.8 Cancelled and Diverted Flights

# In[ ]:


df["Cancelled"].value_counts()


# There were only a tiny fraction of flights that were cancelled.

# In[ ]:


df["Diverted"].value_counts()


# A small proportion of flights were diverted.

# ### 2.9 Bottom 20 origin and dest airports

# In[ ]:


bottom_20_origin = df['Origin'].value_counts()[-20:]
bottom_20_origin


# In[ ]:


bottom_20_dest = df['Dest'].value_counts()[-20:]
bottom_20_dest


# There are some airports that are highly infrequently used.

# Origin Airport Binning Exploration

# In[ ]:


origin = df['Origin'].value_counts()
origin.quantile([.25, .5, .75])


# ### 2.10 Carriers

# In[ ]:


sns.countplot(x='UniqueCarrier', palette='Set2', data=df, order=df['UniqueCarrier'].value_counts().index)
plt.show()

df['UniqueCarrier'].value_counts()


# Despite there being only 20 unique carriers, WN is the most popular by a large margin of approximately 2X the next carrier (AA). As the carrier column is categorical without an ordinal ranking, it can be binned based on the 'popularity' of an airline which gives an ordinal ranking.

# In[ ]:


def label_carrier(carrier):
    if carrier == 'WN':
        return 'highly_popular'  # Highly Popular (> 200,000 occurences)
    if carrier in ['AA', 'MQ', 'UA', 'OO', 'DL', 'XE', 'CO']:
        return 'popular'  # Popular (100,000 - 200,000 occurences)
    if carrier in ['US', 'EV', 'NW', 'FL', 'YV', 'B6', 'OH', '9E', 'AS', 'F9', 'HA', 'AQ']:
        return 'unpopular'  # Unpopular


# In[ ]:


df['carrier_popularity'] = df['UniqueCarrier'].apply(lambda x: label_carrier(x))
sns.countplot(x='carrier_popularity', data=df, order=df['carrier_popularity'].value_counts().index)
plt.show()


# There are approximately 2X more unpopular carriers over the most popular carrier, WN.

# ### 2.11 Delay Reasons Correlation
# Hypothesis: delay reasons may affect each other 
# 
# 

# In[ ]:


corrmat = df[delay_reason_list].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, cmap="crest");
plt.show()


# The chart above shows a weak correlation.

# ## 3. Preprocessing
# 
# Since we are trying to predict if a flight will be delayed before its departure, we consider features that are only available before departure.
# 
# Features not used and why:
# - Year: because the dataset is made up completely from 2008.
# - DepTime: actual departure time is unknown before flight boarding
# - ArrTime: actual arrival time at dest is unknown before flight boarding
# - ActualElapsedTime: unknown before flight boarding
# - AirTime: unknown before flight landing
# - DepDelay: used for target column
# - ArrDelay: unknown before flight landing
# - TaxiOut: unknown before flight boarding
# - Cancelled: unknown before flight check-in, also insignificant data
# - CancellationCode: same as Cancelled
# - Diverted: unknown before flight takeoff

# In[ ]:


# For 2008
days_in_month = {
    1: 31,
    2: 29,
    3: 31,
    4: 30,
    5: 30,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}


# The status quo of 15 minutes as the acceptable on-time performance to classify a flight as ‘late’ is widely accepted in the field currently. We aim to identify flights that will cause great distress to all parties involved, thus we have set a threshold of 30 minutes to determine if a flight is delayed.

# ### 3.1 Train-test split

# In[ ]:


df_flights['DepDel30'] = np.where(df_flights['DepDelay'] > 30, 1, 0)

pre_depart_features = [
    'Month', 'DayofMonth', 'DayOfWeek', 'CRSDepTime', 'CRSArrTime', 'UniqueCarrier', 
    'FlightNum', 'TailNum', 'CRSElapsedTime', 'Origin', 'Dest', 'Distance', 
    'TaxiIn', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 
    'LateAircraftDelay'
]

# Drop nulls
df_flights.dropna(subset=['TailNum', 'CRSElapsedTime'], inplace=True)
df_flights = df_flights[df_flights['TailNum'] != 'Unknow']  # Incorrect row

X = df_flights[pre_depart_features]
y = df_flights['DepDel30']


# In[ ]:


# As there is a class imbalance, stratifying the train test split ensures there is a balance of positive labels in the dataset
# However, the dataset can be considered a time series, thus shuffle=False is an important state of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)

# Since feature engineering is done on the time features, we will use the stratified split


# In[ ]:


print(X_train.shape)
print(y_train.shape)


# In[ ]:


y_train.value_counts()


# Label = 1 is categorised as a delayed flight with an actual departure time of 30 minutes greater than the scheduled departure time. From the training data, it can be observed that there are more 'not delayed' flights. There is no drastic class imbalance present, albeit some. Thus we will consider using the 'balanced' hyperparameter for model training.

# ### 3.2 Train dataset exploration
# for quantile thresholds for binning

# #### 3.2.1 Carrier by popularity  
# This feature will be hardcoded based on the quantile information by counts in the training dataset.

# In[ ]:


X_train_temp = X_train.copy()


# In[ ]:


X_train_temp['UniqueCarrier'].value_counts()


# In[ ]:


X_train_temp['UniqueCarrier'].value_counts().quantile([.25, .5, .75], interpolation='nearest')


# In[ ]:


# Using quantiles for threshold
def label_carrier_train(carrier):
    if carrier in ['WN', 'AA', 'MQ', 'UA', 'OO']:
        return 'popular'
    if carrier in ['DL', 'XE', 'CO', 'US', 'EV', 'NW', 'FL', 'YV', 'B6']:
        return 'average'
    if carrier in ['OH', '9E', 'AS', 'F9', 'HA', 'AQ']:
        return 'unpopular'


# In[ ]:


X_train_temp['carrier_popularity'] = X_train_temp['UniqueCarrier'].apply(lambda x: label_carrier_train(x))
sns.countplot(x='carrier_popularity', data=X_train_temp, order=X_train_temp['carrier_popularity'].value_counts().index)
plt.show()


# We are able to capture the trends of the carriers' usages.

# #### 3.2.2 Origin, Dest
# Create airport type column based on iata_code of airport. This new column will be one-hot encoded

# In[ ]:


def bin_airport(df, left_col='Origin'):
    df = df.copy()
    codes_airports_types = pd.read_csv('/content/code_airport_type.csv', index_col='iata_code')

    df = df.merge(codes_airports_types, left_on=left_col, right_on='iata_code', how='left')
    df.rename({'type': left_col+'_airport_type'}, axis=1, inplace=True)

    return df


# Using the mean departure delay minutes for the originating airport, a mean arrival delay for the destination airport and hardcoding these values may lead to more intricate aspects of the data.

# In[ ]:


train_idx = X_train.index
df_train_idx = df[df.index.isin(train_idx)]
df_train_idx.head(5)


# In[ ]:


origin_avg_time = df_train_idx.groupby('Origin')['DepDelay'].mean().rename('origin_mean_DepDelay')
origin_avg_time.to_csv('/content/origin_avg_time.csv')


# In[ ]:


origin_avg_time = pd.read_csv('/content/origin_avg_time.csv')
origin_avg_time


# In[ ]:


dest_avg_time = df_train_idx.groupby('Dest')['ArrDelay'].mean().rename('dest_mean_ArrDelay')
dest_avg_time.to_csv('/content/dest_avg_time.csv')


# In[ ]:


dest_avg_time = pd.read_csv('/content/dest_avg_time.csv')
dest_avg_time


# #### 3.2.3 Tail Number
# A particular plane identified by its TailNum could have an effect on the Delay. Thus, we deem this feature to be useful for the model. However, it has a high number of categories and thus is highly cardinal.  
# 
# Quantile binning and One-Hot Encoding will not work for this column. Thus we will try to hardcode the mean delay on the training dataset.

# In[ ]:


X_train_temp['TailNum'].value_counts()


# In[ ]:


tailnum_avg_time = df_train_idx.groupby('TailNum')['DepDelay'].mean().rename('tailnum_mean_DepDelay')
tailnum_avg_time.to_csv('/content/tailnum_avg_time.csv')


# In[ ]:


tailnum_avg_time = pd.read_csv('/content/tailnum_avg_time.csv')
tailnum_avg_time


# In[ ]:


def add_mean_column(df, file_path, col_name):
    avg_time_df = pd.read_csv(file_path)
    df = df.merge(avg_time_df, on=col_name, how='left')
    return df


# ### 3.3 Feature Engineering

# In[ ]:


def split_hour(num_t):
    # this time is given as hhmm, thus for any length lesser than 3 digits,
    # we will assume it is during the midnight hour
    s_time = str(num_t)
    if len(s_time) > 2:
        s_min = s_time[-2:]
        s_hour = s_time.replace(s_min, '')
        if s_hour == '':
            s_hour = s_min
    else:
        s_hour = s_time
    return float(s_hour)

def split_min(num_t):
    s_time = str(num_t)
    s_min = s_time[-2:]
    return float(s_min)


def process_features(df):
    df = df.copy()
    # Split time columns into hour and minute
    df['CRSDepHour'] = df['CRSDepTime'].apply(lambda x: split_hour(x))
    df['CRSDepMin'] = df['CRSDepTime'].apply(lambda x: split_min(x))

    df['CRSArrHour'] = df['CRSArrTime'].apply(lambda x: split_hour(x))
    df['CRSArrMin'] = df['CRSArrTime'].apply(lambda x: split_min(x))
    
    # Drop time columns
    df.drop(['CRSDepTime', 'CRSArrTime'], axis=1, inplace=True)

    # Transform days, time in sin & cos
    df['days_in_month'] = df['Month'].map(days_in_month)

    df['Month_sin'] = np.sin(df['Month'] * (2 * np.pi / 12))
    df['Month_cos'] = np.cos(df['Month'] * (2 * np.pi / 12))

    df['DayofMonth_sin'] = np.sin(df['DayofMonth'] * (2 * np.pi / df['days_in_month']))
    df['DayofMonth_cos'] = np.cos(df['DayofMonth'] * (2 * np.pi / df['days_in_month']))

    df['DayOfWeek_sin'] = np.sin(df['DayOfWeek'] * (2 * np.pi / 7))
    df['DayOfWeek_cos'] = np.cos(df['DayOfWeek'] * (2 * np.pi / 7))

    df['CRSDepHour_sin'] = np.sin(df['CRSDepHour'] * (2 * np.pi / 24))
    df['CRSDepHour_cos'] = np.cos(df['CRSDepHour'] * (2 * np.pi / 24))

    df['CRSDepMin_sin'] = np.sin(df['CRSDepMin'] * (2 * np.pi / 60))
    df['CRSDepMin_cos'] = np.cos(df['CRSDepMin'] * (2 * np.pi / 60))

    df['CRSArrHour_sin'] = np.sin(df['CRSArrHour'] * (2 * np.pi / 24))
    df['CRSArrHour_cos'] = np.cos(df['CRSArrHour'] * (2 * np.pi / 24))

    df['CRSArrMin_sin'] = np.sin(df['CRSArrMin'] * (2 * np.pi / 60))
    df['CRSArrMin_cos'] = np.cos(df['CRSArrMin'] * (2 * np.pi / 60))

    # Create is_weekend col
    df['is_weekend'] = df['DayOfWeek'].isin([1, 7]).astype(int)

    # Drop day and time columns
    df.drop(['Month', 'DayofMonth', 'DayOfWeek', 'CRSDepHour', 'CRSDepMin', 'CRSArrHour', 'CRSArrMin', 'days_in_month'], axis=1, inplace=True)

    # Log Distance column as there is a high std dev
    df['log_Distance'] = np.log(df['Distance'])

    # Bin and OHE Carriers
    df['carrier_popularity'] = df['UniqueCarrier'].apply(lambda x: label_carrier_train(x))
    df = pd.get_dummies(df, prefix=['carrier'], columns=['carrier_popularity'])

    # Bin and OHE Origin Airport
    df = bin_airport(df)
    df = pd.get_dummies(df, prefix=['Origin'], columns=['Origin_airport_type'])

    # Bin and OHE Dest Airport
    df = bin_airport(df, left_col='Dest')
    df = pd.get_dummies(df, prefix=['Dest'], columns=['Dest_airport_type'])

    # Get avg delay
    df = add_mean_column(df, '/content/origin_avg_time.csv', 'Origin')
    df = add_mean_column(df, '/content/dest_avg_time.csv', 'Dest')
    df = add_mean_column(df, '/content/tailnum_avg_time.csv', 'TailNum')

    # Drop original columns
    df.drop(['Distance', 'UniqueCarrier', 'Origin', 'Dest', 'TailNum'], axis=1, inplace=True)  #, 'TailNum'

    # Impute nulls with 0
    for col in ['TaxiIn', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'dest_mean_ArrDelay', 'tailnum_mean_DepDelay']:
        df[col] = df[col].fillna(0)

    return df

X_pr = process_features(X_train.copy())
X_pr.head(5)


# ## 4. Modelling

# In[ ]:


transform_features = FunctionTransformer(process_features)


# In[ ]:


models_dict = {
    "log_reg": LogisticRegression(class_weight='balanced', random_state=123),
    "svm": LinearSVC(class_weight='balanced', random_state=123),
    "dtc": DecisionTreeClassifier(class_weight='balanced', random_state=123),
    "rfc": RandomForestClassifier(class_weight='balanced', random_state=123),
}


# In[ ]:


def create_pipeline(classifier):
    pipe = make_pipeline(
        clone(transform_features), 
        MinMaxScaler(), 
        classifier
    )
    return pipe


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntrained_pipes = {}\nfor key, classifier in models_dict.items():\n    print(f"Model: {classifier}\\n")\n\n    pipe = create_pipeline(classifier)\n    # fit the pipeline on the transformed data\n    pipe.fit(X_train.copy(), y_train)\n\n    trained_pipes[key] = pipe\n\n    # make predictions\n    y_pred = pipe.predict(X_test.copy())\n\n    # evaluate pipeline\n    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)\n    cv_scores = cross_val_score(\n        pipe, \n        X_train.copy(), \n        y_train, \n        scoring=\'f1\', \n        cv=cv, \n        n_jobs=-1\n    )\n    print(f"Cross validated mean F1 Score = {round(np.mean(cv_scores), 3)}")\n\n    print(f"Balanced Accuracy on Test set: {round(balanced_accuracy_score(y_test, y_pred), 3)}")\n    print(f"F1 Score on Test set: {round(f1_score(y_test, y_pred), 3)}\\n")\n\n    print(f"Confusion Matrix:")\n    plot_confusion_matrix(pipe, X_test.copy(), y_test, labels=[1, 0])  \n    plt.show()\n\n    print(f"\\nClassification report: \\n{classification_report(y_test, y_pred, labels=[1, 0])}")\n    print("----------------------------------------------------------------")')


# The meaning of each of the labels is as follows:
# - 0: a flight is not delayed
# - 1: a flight is delayed  
# (Where a delay is defined as the actual departure time of a flight to be greater than 30 minutes from its scheduled departure time.)
# 
# As the class = 1 is a minority class, and is the class we are interested to predict, we observe the evaluation scores for this class in particular.  
# 
# Exploring the cross validation scores among the trained classifiers, the Random Forest Classifier (RFC) has the highest score at 0.93. 
# 
# Based the test dataset, the RFC also has the best performance with the highest F1 score for the positive class. It also has the highest Balanced Accuracy on the test dataset.

# In[ ]:


trained_pipes


# In[ ]:


# save all pipelines
joblib.dump(trained_pipes, '/content/trained_pipes.pkl')


# In[ ]:


# save best model a file
joblib.dump(trained_pipes['rfc'], '/content/rf_clf.pkl')


# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)


# In[ ]:


get_ipython().system("cp trained_pipes.pkl '/content/gdrive/MyDrive'")


# In[ ]:


get_ipython().system("cp rf_clf.pkl '/content/gdrive/MyDrive'")


# ## 5. Evaluation

# In[ ]:


# load saved model
# rfc_clf = joblib.load('/content/gdrive/MyDrive/rf_clf.pkl')


# In[ ]:


# y_pred_rfc = rfc_clf.predict(X_test.copy())


# ### 5.1 Precision-Recall Curve
# Of the Random Forest Classifier with class = 1 (flight delayed)

# In[ ]:


predictions = trained_pipes["rfc"].predict_proba(X_test.copy())[:, 1]
precision, recall, _ = precision_recall_curve(y_test, predictions)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.show()


# ### 5.2 ROC Curve
# Of the Random Forest Classifier with class = 1 (flight delayed)

# In[ ]:


# Draw the ROC
RocCurveDisplay.from_estimator(trained_pipes["rfc"], X_test.copy(), y_test, pos_label=1)


# In[ ]:


# Calculate AUCROC Score
round(roc_auc_score(y_test, predictions), 3)


# The curves above in 5.1 and 5.2 are in relation to the positive class (flight delays) of the random forest classifier. As False Negatives are highly costly in this context, we aim to increase recall. In these graphs, the Recall and TPR have the same meaning. Thus it can be viewed from the PR and ROC curves that the model performs well.

# ### 5.3 Feature Importances
# Of Random Forest Classifier

# In[ ]:


feature_names = [
    'FlightNum','CRSElapsedTime','TaxiIn','CarrierDelay','WeatherDelay','NASDelay',
    'SecurityDelay','LateAircraftDelay','Month_sin','Month_cos','DayofMonth_sin',
    'DayofMonth_cos','DayOfWeek_sin','DayOfWeek_cos','CRSDepHour_sin','CRSDepHour_cos',
    'CRSDepMin_sin','CRSDepMin_cos','CRSArrHour_sin','CRSArrHour_cos','CRSArrMin_sin',
    'CRSArrMin_cos','is_weekend','log_Distance','carrier_average','carrier_popular',
    'carrier_unpopular','Origin_large_airport','Origin_medium_airport','Origin_small_airport',
    'Dest_large_airport','Dest_medium_airport','Dest_small_airport','origin_mean_DepDelay',
    'dest_mean_ArrDelay','tailnum_mean_DepDelay'
]


# In[ ]:


importances = trained_pipes["rfc"]["randomforestclassifier"].feature_importances_
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()


# We extracted the feature importances for the RFC and noticed that the time based transformed features are important. The remaining transformed features also contribute to the model's predictions albeit at a lower rate. 
# 
# The delay propagation due to late aircraft delay is the most important feature because one late arriving carrier is the direct cause of the following departure delay.  
# 
# CarrierDelay being within the control of the air carrier due to awaiting passengers, crew, slow boarding, and aircraft inspections is the second most important feature. This rationale makes for a practical understanding that a flight can be potentially delayed.

# ### 5.4. Business Application
# In the context of this project, False Negatives (FN) mean that there is an actual delay that was not predicted, this will have the highest cost as shown by the cost benefit matrix below. Thus as FNs are highly costly, we aim to have a high recall score for the positive class.

# #### 5.4.1 Expected Value

# In[ ]:


cost_benefit_matrix = np.array([
    [16.6, 81.3],
    [0.5, 0]
])


# In[ ]:


for key in trained_pipes:
    # Extract the model from the dictionary
    model = trained_pipes[key]
    
    # Compute predictions
    y_pred = model.predict(X_test)
    
    # Compute confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred, labels=[1, 0], normalize='all')
    
    # Compute expected value
    expected_val = (conf_mat * cost_benefit_matrix).sum()
    
    print(key, "\t", round(expected_val, 3))


# As we are determining the price of the cost incurred with the cost-benefit matrix, we aim to minimize the expected value.  
# Thus, the random forest classifier has the lowest expected cost value while the logistic regression has the highest.

# #### 5.4.2 Cost Curve

# In[ ]:


y_scores = trained_pipes["rfc"].predict_proba(X_test.copy())[:, 1]


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test, y_scores)


# In[ ]:


expected_values = []
for t in thresholds:
    # Compute predictions at that threshold
    y_pred = [1 if score > t else 0 for score in y_scores]

    # Compute confusion matrix for those predictions
    mat = confusion_matrix(y_test, y_pred, labels=[1,0], normalize='all')
    
    # Calculate expected value
    expected_value = (mat * cost_benefit_matrix).sum()
    
    # Add to list
    expected_values.append(expected_value)

plt.plot(thresholds, expected_values)
plt.xlabel('Class Probabilities')
plt.ylabel('Expected Value')


# A maximum expected cost value of 35 is achieved with a class probability of between 1.0 and 2.0 for the random forest classifier. It can be observed that despite class probabilities being at 0, there is a minimum cost incurred. The expected values of 9.5 and 10.76 are relatively close as compared to the maximum. Thus the threshold can be adjusted to a lower value to reduce the expected cost.

# ## 6. Hyperparameter Tuning
# We utilize the HalvingGridSearchCV to speed up the hyperparameter tuning selection.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nparam_grid = {\n    \'randomforestclassifier__n_estimators\': [50, 100, 200],\n    \'randomforestclassifier__max_features\': [\'sqrt\', \'log2\'],\n    \'randomforestclassifier__max_depth\': [3, 5, 8],\n    \'randomforestclassifier__min_samples_split\': [2, 5, 10],\n}\n\npipeline_rfc = make_pipeline(\n        transform_features, \n        MinMaxScaler(), \n        RandomForestClassifier(class_weight=\'balanced\', random_state=123),\n)\n\nfolds = StratifiedKFold(\n    n_splits=3, \n    shuffle=True, \n    random_state=123\n)\n\ngrid = HalvingGridSearchCV(\n    pipeline_rfc, \n    param_grid, \n    scoring="f1", \n    cv=folds, \n    random_state=123, \n    n_jobs=-1, \n    verbose=2\n)\n\n# fitting the model for grid search\ngrid.fit(X_train.copy(), y_train)')


# In[ ]:


# Finding best hyperparameters
print(f"Best Hyperparameters: \n{grid.best_params_}\n")
print(f"Mean CV Score of the best estimator: {round(grid.best_score_, 3)}")


# In[ ]:


# Score on test dataset
grid_predictions = grid.predict(X_test.copy())
print(
    f"Grid Search F1 Score on Test data: {round(f1_score(y_test, grid_predictions), 3)}"
)


# In[ ]:


print(classification_report(y_test, grid_predictions, labels=[1, 0]))


# In[ ]:


plot_confusion_matrix(grid, X_test.copy(), y_test, labels=[1, 0])


# The use of HalvingGridSearchCV for hyperparameter tuning led to a decrease in the mean CV score on the training data and F1 score and Balanced Accuracy on the test data. In addition, the Precision and Recall scores for both classes decreased from the default model. The numbers of TP, TN, and FP decreased while the FN increased marginally. To evaluate this tuned model, we will calculate its expected value.

# In[ ]:


# Save fine-tuned model
joblib.dump(grid, '/content/rfc_hpt.pkl')


# In[ ]:


get_ipython().system("cp rfc_hpt.pkl '/content/gdrive/MyDrive'")


# **Expected Value**

# In[ ]:


# load hpt
# rfc_hpt = joblib.load('/content/gdrive/MyDrive/rfc_hpt.pkl')


# In[ ]:


# grid_predictions = rfc_hpt.predict(X_test.copy())


# In[ ]:


conf_mat_rfc_hpt = confusion_matrix(
    y_test, grid_predictions, 
    labels=[1, 0],
    normalize='all'
)
conf_mat_rfc_hpt


# In[ ]:


round((conf_mat_rfc_hpt * cost_benefit_matrix).sum(), 2)


# Despite having slightly lower evaluation scores, the expected value for the hyperparameter tuned model is much higher thus indicating it delivers a substandard business application. The rationale for this is that the number of False Negatives increased through the hyperparameter tuning and the False Negatives have the highest associated cost. In view of this, the default RFC model offers sufficient business value.
