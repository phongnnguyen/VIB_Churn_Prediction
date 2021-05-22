import pandas as pd
import numpy as np
from datetime import datetime

class Processing_Data:
    def __init__(self,DataPath):
        self.DataPath = DataPath


class Define_Churn_Users:
    '''Define what are churned users'''
    def __init__(self, df_activity):
        self.df_activity = df_activity

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

    def highest_day_inactive(self,init_noDate = None):
        if init_noDate is not None:
            print('Here')


    def label_churn_for_user(self):
        print('here')
