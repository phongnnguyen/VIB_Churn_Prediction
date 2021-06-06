import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn import metrics
from sklearn import preprocessing
import pickle
def hour_to_shift(series):
    shift = []
    for v in series:
        if 5 >= v >= 0:
            shift.append(1)
        elif 5 > v >= 12:
            shift.append(2)
        elif 12 > v >= 18:
            shift.append(3)
        else:
            shift.append(4)
    return shift

def day_of_week_num(series):
    DOW = []
    for v in series:
        if v == 'Mon':
            DOW.append(1)
        elif v == 'Tue':
            DOW.append(2)
        elif v == 'Wed':
            DOW.append(3)
        elif v == 'Thu':
            DOW.append(4)
        elif v == 'Fri':
            DOW.append(5)
        elif v == 'Sat':
            DOW.append(6)
        else:
            DOW.append(7)
    return DOW

def EB_register_chanel(series):
    ERC = []
    for v in series:
        if v == 'BRANCH':
            ERC.append(1)
        elif v == 'MYVIB':
            ERC.append(2)
        elif v == 'AUTO-JOB':
            ERC.append(3)
        else:
            ERC.append(4)
    return ERC
def standardize(df):
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return x_scaled

def KNN_model(X_train, y_train, X_test, y_test, save = True, plotting = False):
    from sklearn.neighbors import KNeighborsClassifier
    error_rate = []
    # Will take some time
    for i in range(1, 30):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))

    knn = KNeighborsClassifier(n_neighbors=np.argmin(error_rate))
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    if save:
        KNNPickle = open('model/KNN_model', 'wb')
        pickle.dump(knn, KNNPickle)
    # Model Accuracy, how often is the classifier correct?
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    if plotting:
        plt.figure(figsize=(10,6))
        plt.plot(range(1,30), error_rate, color='blue', linestyle='dashed', marker='o',
                 markerfacecolor='red', markersize=10)
        plt.title('Error Rate vs. K Value')
        plt.xlabel('K')
        plt.ylabel('Error Rate')

    return accuracy


def Logistic_model(X_train, y_train, X_test, y_test, save = True):
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression()
    # fit the model with data
    logreg.fit(X_train,y_train)
    y_pred=logreg.predict(X_test)
    # # Model Accuracy, how often is the classifier correct?
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    if save:
        LGTPickle = open('model/Logistic_model', 'wb')
        pickle.dump(logreg, LGTPickle)
        # load the model from disk
        # loaded_model = pickle.load(open('Logistic_model', 'rb'))
        # result = loaded_model.predict(X_test)
    return accuracy

def XGBOOST_model(X_train, y_train, X_test, y_test, save = True):
    from xgboost import XGBClassifier
    # fit model no training data
    model = XGBClassifier(learning_rate = 0.1, n_estimators=1000, max_depth=5,
                           min_child_weight=1, gamma=0,subsample=0.8,
                           colsample_bytree=0.8, objective = 'binary:logistic',
                           nthread=4, scale_pos_weight=1, seed=27)
    model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = metrics.accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    if save:
        XGBPickle = open('model/XGBOOST_model', 'wb')
        pickle.dump(model, XGBPickle)
    return accuracy

# # Preprocessing data befor training
df = pd.read_csv('data/combine_information.csv')
df = df.drop('CUSTOMER_NUMBER.1',axis=1)
# remove extreme Age

df = df[(df['AGE']>15) & (df['AGE']<80)]
df = df.fillna(0)
df['ACTIVITY_HOUR'] = hour_to_shift(df['ACTIVITY_HOUR'])
df['DAY_OF_WEEK'] = day_of_week_num(df['DAY_OF_WEEK'])
df['CLIENT_SEX'] = [1 if v == 'M' else 0 for v in df['CLIENT_SEX']]
df['EB_REGISTER_CHANNEL'] = EB_register_chanel(df['EB_REGISTER_CHANNEL'])

df.index = df['CUSTOMER_NUMBER']
df_nonchurn = df[df['LABEL']==0]
df_churn = df[df['LABEL'] == 1]
df_nonchurn_sample = df_nonchurn.sample(n=len(df_churn))
df_final = pd.concat([df_nonchurn_sample, df_churn])
df_final = shuffle(df_final)

# Training
X = df_final[['numAct','AGE','CLIENT_SEX','EB_REGISTER_CHANNEL',
                        'DAY_OF_WEEK','ACTIVITY_HOUR','TRANS_AMOUNT']]
Y = df_final['LABEL']

# Split train test data set
X_train, X_test, y_train, y_test = \
    train_test_split(X, Y, test_size=0.3) # 70% training and 30% test

# Run model
# accuracy = KNN_model(X_train,y_train,X_test,y_test)
# accuracy = Logistic_model(X_train,y_train,X_test,y_test)
accuracy = XGBOOST_model(X_train,y_train,X_test,y_test)


