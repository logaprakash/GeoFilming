#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:28:47 2018

@author: sakshi
"""

"""
Cleaning Data
"""

import pandas as pd
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import zipcode
from uszipcode import ZipcodeSearchEngine

"""
Reading the data
"""
users_df = pd.read_csv('/home/sakshi/Documents/ml-1m/users.dat',delimiter="::",engine="python",header=None, names=["UserID","Gender","Age","Occupation","Zipcode"],index_col="UserID")
movies_df = pd.read_csv('/home/sakshi/Documents/ml-1m/movies.dat',delimiter="::",engine="python",header=None, names=["MovieID","Title","Genres"])
ratings_df = pd.read_csv('/home/sakshi/Documents/ml-1m/ratings.dat',delimiter="::",engine="python",header=None, names=["UserID","MovieID","Rating","Timestamp"])

"""
Cleaning
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
movies_df_test = pd.concat([movies_df,genres_encoded],axis=1,join='inner')

