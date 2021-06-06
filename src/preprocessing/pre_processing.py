import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


class Processing_Data:
    def __init__(self, ldYear, df_activity, df_transaction, df_customer):

        self.df_activity = df_activity
        self.df_customer = df_customer
        self.df_transaction = df_transaction
        self.ldYear = ldYear
    def remove_outlier(self):
        print('here')

    def remove_duplicate(self):
        print('here')

    def reindex_dataframe(self,df):
        df.index = df['CUSTOMER_NUMBER']
        return df

    def hour_to_shift(self, series):
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

    def combine_data_from_multiple_table(self):
        df_merging_activity = self.df_activity.groupby(['CUSTOMER_NUMBER'])[
            'ACTIVITY_NAME'].apply(','.join).reset_index()
        df_merging_activity = self.reindex_dataframe(df_merging_activity)
        # Calculate number of activity
        df_merging_activity['numAct'] = [len(set(ele.split(","))) for ele in
                                         df_merging_activity['ACTIVITY_NAME']]
        # group activity for each customer
        # df_merging_activity['listAct'] = [ele.split(",") for ele in
        #                                   df_merging_activity['ACTIVITY_NAME']]
        # calculate age
        # fix wrong DOB
        VIB_customer_resINDX = self.reindex_dataframe(self.df_customer)
        VIB_customer_resINDX.at[273643, 'DATE_OF_BIRTH'] = \
            '1996-07-28 00:00:00'
        age_cal = pd.DataFrame((self.ldYear - pd.to_datetime(
            VIB_customer_resINDX.loc[df_merging_activity['CUSTOMER_NUMBER']][
                'DATE_OF_BIRTH'])).astype('<m8[Y]'))
        age_cal.columns = ['AGE']

        df_merging_activity = df_merging_activity.join(age_cal)  # Join age
        # merging information of customer
        df_merging_activity = df_merging_activity.join(
            VIB_customer_resINDX.loc[df_merging_activity['CUSTOMER_NUMBER']][
                ['DATE_OF_BIRTH', 'CLIENT_SEX', 'STAFF_VIB',
                 'EB_REGISTER_CHANNEL', 'SMS', 'VERIFY_METHOD']])
        # add list of activity
        # # list of activity
        # df_activity_stat = df_merging_activity.copy()
        # df_activity_stat.index = df_activity_stat['CUSTOMER_NUMBER']
        # df_merging_activity = df_merging_activity.join(
        #     df_activity_stat[['numAct', 'listAct']])
        df_merging_activity = df_merging_activity.drop(['ACTIVITY_NAME'],axis=1)
        df_merging_activity = \
            df_merging_activity.join(self.df_activity.
                                     groupby(['CUSTOMER_NUMBER']).agg(
            lambda x: x.value_counts().index[0])
                                     [['DAY_OF_WEEK','ACTIVITY_HOUR']])
        # merging transaction
        groupby_transac = \
            self.df_transaction.groupby(['CUSTOMER_NUMBER']).sum()

        df_merging_activity = \
            df_merging_activity.join(groupby_transac['TRANS_AMOUNT'])

        return df_merging_activity



class Define_Churn_Users:
    '''Define what are churned users'''

    def __init__(self, df_activity, endDate):
        self.df_activity = df_activity
        self.endDate = endDate

    def calculate_number_inactive_all_users(self):
        '''Calculate the inactive day between 2 active day for each user'''
        df_groupby_customer_date = self.df_activity.groupby(
            ['CUSTOMER_NUMBER', 'ACTIVITY_DATE']).mean()
        df_groupby_customer_date[
            'GROUPBY_DATE'] = df_groupby_customer_date.index.get_level_values(
            'ACTIVITY_DATE')
        df_groupby_customer_date[
            'GROUPBY_CUSTOMER'] = df_groupby_customer_date.index.get_level_values(
            'CUSTOMER_NUMBER')

        format_date = [datetime.strptime(td, '%Y-%m-%d') for td in
                       df_groupby_customer_date['GROUPBY_DATE']]
        df_groupby_customer_date['GROUPBY_DATE'] = format_date

        # Calculate number of inactive

        number_of_inactive = \
        df_groupby_customer_date.groupby(['GROUPBY_CUSTOMER'])[
            'GROUPBY_DATE'].diff() / np.timedelta64(1, 'D')
        number_of_inactive = pd.DataFrame(number_of_inactive - 1)

        return number_of_inactive

    def calculate_number_inactive_from_last_purchase(self):
        last_day_of_year = self.endDate
        # get last purchase date
        last_purchase_customer = self.df_activity.sort_values(
            by="ACTIVITY_DATE").drop_duplicates(
            subset=["CUSTOMER_NUMBER"], keep="last")
        last_purchase_customer.index = last_purchase_customer[
            'CUSTOMER_NUMBER']

        # Calculate number inactive day from last purchase of user
        last_purchase_customer['LAST_PURCHASE'] = \
            (last_day_of_year - pd.to_datetime(last_purchase_customer[
                                                'ACTIVITY_DATE'])).dt.days - 1
        last_purchase_customer['LAST_PURCHASE'][
            last_purchase_customer['LAST_PURCHASE'] < 0] = 0

        return last_purchase_customer

    def label_churn_for_user(self, last_purchase_day_num, init_noDate=10,
                             filename='Combine_Data_VIB.csv'):
        # Initial label churn or not churn
        last_purchase_day_num['LABEL'] = 0
        last_purchase_day_num['LABEL'][
            last_purchase_day_num['LAST_PURCHASE'] > 12] = 1

        S = self.endDate - timedelta(days=12)
        df_before_S = self.df_activity[
            pd.to_datetime(self.df_activity['ACTIVITY_DATE'],
                           infer_datetime_format=True) < S]

        last_purchase_before_S = df_before_S.sort_values(
            by="ACTIVITY_DATE").drop_duplicates(subset=["CUSTOMER_NUMBER"],
                                                keep="last")

        d = range(1, 30)
        total_error = []
        for i in d:
            Sd = S - timedelta(days=i)
            purchase_before_Sd = last_purchase_before_S[
                pd.to_datetime(last_purchase_before_S['ACTIVITY_DATE'],
                               infer_datetime_format=True) < Sd]

            purchase_between_Sd = last_purchase_before_S[
                (pd.to_datetime(last_purchase_before_S['ACTIVITY_DATE'],
                                infer_datetime_format=True) >= Sd)
                & (pd.to_datetime(last_purchase_before_S['ACTIVITY_DATE'],
                                  infer_datetime_format=True) < S)]
            count_label = \
            last_purchase_day_num.loc[purchase_before_Sd['CUSTOMER_NUMBER']][
                'LABEL'].value_counts()
            label_nonchurn = count_label[0]

            count_label_between = \
            last_purchase_day_num.loc[purchase_between_Sd['CUSTOMER_NUMBER']][
                'LABEL'].value_counts()
            try:
                label_churn_between = count_label_between[1]
            except:
                label_churn_between = 0

            epsilon_FP = label_nonchurn / \
                         last_purchase_day_num.loc[
                             last_purchase_before_S['CUSTOMER_NUMBER']][
                             'LABEL'].value_counts()[0]
            epsilon_FN = label_churn_between / \
                         last_purchase_day_num.loc[
                             last_purchase_before_S['CUSTOMER_NUMBER']][
                             'LABEL'].value_counts()[1]

            total_error.append(0.5 * epsilon_FP + 0.5 * epsilon_FN)

        highest_number_of_inactive = np.argmin(np.array(total_error))

        # re-lable with new threshold

        last_purchase_day_num['LABEL'] = 0
        last_purchase_day_num['LABEL'][
            last_purchase_day_num[
                'LAST_PURCHASE'] > init_noDate + highest_number_of_inactive] = 1
        last_purchase_day_num.to_csv('data/' + filename)
        return last_purchase_day_num
