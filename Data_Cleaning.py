#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:28:47 2018

@author: sakshi
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from uszipcode import ZipcodeSearchEngine

"""
Reading the data
"""
users_df.columns = pd.read_csv('DataSet/users.dat',delimiter="::",engine="python",header=None, names=["UserID","Gender","Age","Occupation","Zipcode"],index_col="UserID")
movies_df = pd.read_csv('DataSet/movies.dat',delimiter="::",engine="python",header=None, names=["MovieID","Title","Genres"])
ratings_df = pd.read_csv('DataSet/ratings.dat',delimiter="::",engine="python",header=None, names=["UserID","MovieID","Rating","Timestamp"])

"""
Cleaning and Preprocessing
"""
users_df["Altered_Zipcode"] = users_df.Zipcode.apply(lambda x : x[:5])
search = ZipcodeSearchEngine()
users_df["Altered_Zipcode_State"] = users_df.Altered_Zipcode.apply(lambda x : search.by_zipcode(x)["State"])
users_df["Altered_Zipcode_State"]  = users_df.Altered_Zipcode_State.apply(lambda x : 'UX' if type(x) != str else x)

movies_df["Release_Year"] = movies_df.Title.apply(lambda x : int(x[-5:-1]))
movies_df["Title"] = movies_df.Title.apply(lambda x : x[:-7])
movies_df["Genres"] = movies_df.Genres.apply(lambda x : x.split('|'))
vect = CountVectorizer()
genres_list = vect.fit_transform(movies_df.Genres.str.join(' '))
genres_encoded = pd.DataFrame(genres_list.toarray(), columns = vect.get_feature_names())

movies_df_encoded = pd.concat([movies_df,genres_encoded],axis=1,join='inner')
data_df = pd.concat([movies_df.MovieID,genres_encoded],axis=1,join='inner')

states_list = vect.fit_transform(users_df.Altered_Zipcode_State)
states_encoded = pd.DataFrame(states_list.toarray(), columns = vect.get_feature_names())
states_encoded["UserID"] = users_df.index

set_df = ratings_df.drop(columns=["Timestamp"])
set_df_test = pd.merge(left=set_df, right=data_df, how='left', left_on='MovieID',right_on='MovieID')
set_df_test["UserState"] = set_df_test.UserID.apply(lambda x : users_df.Altered_Zipcode_State[x])
set_df_test_2 = pd.merge(left=set_df_test, right=states_encoded, how='left', left_on = 'UserID', right_on = 'UserID')

#a = pd.value_counts(users_df.Altered_Zipcode_State)

dataset_df = set_df_test.drop(columns=["UserID","UserState","MovieID"])
dataset_df_2 = set_df_test_2.drop(columns=["UserID","UserState","MovieID"])

dataset_df.to_pickle('dataset_df.p')
dataset_df_2.to_pickle('dataset_df_states_encoded.p')
