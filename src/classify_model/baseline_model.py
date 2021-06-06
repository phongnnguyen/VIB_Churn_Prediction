import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
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

# processing data before training

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

X_train, X_test, y_train, y_test = \
    train_test_split(X, Y, test_size=0.3) # 70% training and 30% test

# Build model classify

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


