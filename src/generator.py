from src.preprocessing.pre_processing import Define_Churn_Users
import pandas as pd
from datetime import datetime

if __name__ == '__main__':
    df_activity = pd.read_csv('data/Activity.csv')
    df_activity = df_activity.iloc[:10000,:]
    endDate = datetime.strptime(max(df_activity['ACTIVITY_DATE']), '%Y-%m-%d')
    DefineChurn = Define_Churn_Users(df_activity,endDate)
    number_of_inactive = DefineChurn.calculate_number_inactive_all_users()
    df_last_purchase = DefineChurn.calculate_number_inactive_from_last_purchase()
    df_factor_and_label = DefineChurn.label_churn_for_user(last_purchase_day_num = df_last_purchase, filename='test_combine.csv')