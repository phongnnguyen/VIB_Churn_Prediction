from preprocessing.pre_processing import Define_Churn_Users
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
if __name__ == '__main__':
    df_activity = pd.read_csv('data/Activity.csv')
    endDate = datetime.strptime(max(df_activity['ACTIVITY_DATE']), '%Y-%m-%d')
    DefineChurn = Define_Churn_Users(df_activity,endDate)
    number_of_inactive = DefineChurn.calculate_number_inactive_all_users()