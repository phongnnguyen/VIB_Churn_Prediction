from src.preprocessing.pre_processing import Define_Churn_Users, Processing_Data
import pandas as pd
from datetime import datetime

if __name__ == '__main__':
    df_activity = pd.read_csv('data/Activity.csv')
    df_customer = pd.read_csv('data/1.Data_Customer.csv')
    df_transaction = pd.read_csv('data/2.Data_MyVIB_Transaction.csv')
    df_activity = df_activity.iloc[:10000,:]

    endDate = datetime.strptime(max(df_activity['ACTIVITY_DATE']), '%Y-%m-%d')
    LABEL = False
    if LABEL:
        DefineChurn = Define_Churn_Users(df_activity,endDate)
        number_of_inactive = DefineChurn.calculate_number_inactive_all_users()
        df_last_purchase = \
            DefineChurn.calculate_number_inactive_from_last_purchase()
        df_factor_and_label = DefineChurn.label_churn_for_user(
            last_purchase_day_num = df_last_purchase,
            filename='Combine_Data_VIB.csv')
    else:
        df_factor_and_label = pd.read_csv('data/Combine_Data_VIB.csv')
        ProcessingData = Processing_Data(endDate,
                                         df_activity,
                                         df_customer = df_customer,
                                         df_transaction = df_transaction)
        df_combine_information = \
            ProcessingData.combine_data_from_multiple_table()

        df_combine_information = df_combine_information.join(df_factor_and_label['LABEL'])
        df_combine_information.to_csv('data/' + 'combine_information.csv')