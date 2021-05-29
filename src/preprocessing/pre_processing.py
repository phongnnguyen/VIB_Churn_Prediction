import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

class Processing_Data:
    def __init__(self,DataPath):
        self.DataPath = DataPath
    def remove_outlier(self):
        print('here')
    def remove_duplicate(self):
        print('here')
    def combine_data_from_multiple_table(self):
        print('here')

class Define_Churn_Users:
    '''Define what are churned users'''
    def __init__(self, df_activity,endDate):
        self.df_activity = df_activity
        self.endDate = endDate

    def calculate_number_inactive_all_users(self):
        '''Calculate the inactive day between 2 active day for each user'''
        df_groupby_customer_date = self.df_activity.groupby(['CUSTOMER_NUMBER', 'ACTIVITY_DATE']).mean()
        df_groupby_customer_date['GROUPBY_DATE'] = df_groupby_customer_date.index.get_level_values('ACTIVITY_DATE')
        df_groupby_customer_date['GROUPBY_CUSTOMER'] = df_groupby_customer_date.index.get_level_values('CUSTOMER_NUMBER')

        format_date = [datetime.strptime(td, '%Y-%m-%d') for td in df_groupby_customer_date['GROUPBY_DATE']]
        df_groupby_customer_date['GROUPBY_DATE'] = format_date

        # Calculate number of inactive

        number_of_inactive = df_groupby_customer_date.groupby(['GROUPBY_CUSTOMER'])['GROUPBY_DATE'].diff() / np.timedelta64(1, 'D')
        number_of_inactive = pd.DataFrame(number_of_inactive - 1)

        return number_of_inactive

    def calculate_number_inactive_from_last_purchase(self):
        last_day_of_year = self.endDate
        # get last purchase date
        last_purchase_customer = self.df_activity.sort_values(by="ACTIVITY_DATE").drop_duplicates(
            subset=["CUSTOMER_NUMBER"], keep="last")
        last_purchase_customer.index = last_purchase_customer['CUSTOMER_NUMBER']

        # Calculate number inactive day from last purchase of user
        last_purchase_customer['LAST_PURCHASE'] = (last_day_of_year - pd.to_datetime(
            last_purchase_customer['ACTIVITY_DATE'])).dt.days - 1
        last_purchase_customer['LAST_PURCHASE'][last_purchase_customer['LAST_PURCHASE'] < 0] = 0

        return last_purchase_customer


    def label_churn_for_user(self,last_purchase_day_num, init_noDate = 10,filename = 'Combine_Data_VIB.csv'):
        # Initial label churn or not churn
        last_purchase_day_num['LABEL'] = 0
        last_purchase_day_num['LABEL'][last_purchase_day_num['LAST_PURCHASE'] > 12] = 1

        S = self.endDate - timedelta(days=12)
        df_before_S =  self.df_activity[pd.to_datetime(self.df_activity['ACTIVITY_DATE'], infer_datetime_format=True) < S]

        last_purchase_before_S = df_before_S.sort_values(by="ACTIVITY_DATE").drop_duplicates(subset=["CUSTOMER_NUMBER"],
                                                                                             keep="last")

        d = range(1, 30)
        total_error = []
        for i in d:
            Sd = S - timedelta(days=i)
            purchase_before_Sd = last_purchase_before_S[pd.to_datetime(last_purchase_before_S['ACTIVITY_DATE'],
                                                                       infer_datetime_format=True) < Sd]

            purchase_between_Sd = last_purchase_before_S[(pd.to_datetime(last_purchase_before_S['ACTIVITY_DATE'],
                                                                         infer_datetime_format=True) >= Sd)
                                                         & (pd.to_datetime(last_purchase_before_S['ACTIVITY_DATE'],
                                                                           infer_datetime_format=True) < S)]
            count_label = last_purchase_day_num.loc[purchase_before_Sd['CUSTOMER_NUMBER']]['LABEL'].value_counts()
            label_nonchurn = count_label[0]

            count_label_between = last_purchase_day_num.loc[purchase_between_Sd['CUSTOMER_NUMBER']]['LABEL'].value_counts()
            try:
                label_churn_between = count_label_between[1]
            except:
                label_churn_between = 0


            epsilon_FP = label_nonchurn / \
                         last_purchase_day_num.loc[last_purchase_before_S['CUSTOMER_NUMBER']]['LABEL'].value_counts()[0]
            epsilon_FN = label_churn_between / \
                         last_purchase_day_num.loc[last_purchase_before_S['CUSTOMER_NUMBER']]['LABEL'].value_counts()[1]

            total_error.append(0.5 * epsilon_FP + 0.5 * epsilon_FN)

        highest_number_of_inactive = np.argmin(np.array(total_error))

        # re-lable with new threshold

        last_purchase_day_num['LABEL'] = 0
        last_purchase_day_num['LABEL'][
            last_purchase_day_num['LAST_PURCHASE'] > init_noDate + highest_number_of_inactive] = 1
        last_purchase_day_num.to_csv('data/'+filename)
        return last_purchase_day_num

