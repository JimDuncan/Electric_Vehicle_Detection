import pandas as pd
import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
import scipy.stats as scs
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score,precision_score
from sklearn.metrics import precision_score,accuracy_score, recall_score
from itertools import izip
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
import pickle


#pull raw data from csv's into dataframes
def get_data():
    df = pd.read_csv('data_files/EV_train.csv')
    df_labels = pd.read_csv('data_files/EV_train_labels.csv')
    df_final_test= pd.read_csv('data_files/EV_test.csv')
    return df,df_labels,df_final_test


#identifying which households owned EV's
def owns_EV(df_labels):
    df2_labels = df_labels.copy()
    if 'House ID' in df2_labels:
        House_ID = df2_labels['House ID']
        df2_labels.drop('House ID', inplace = True, axis =1)
        house_index = []
        house_index = [row[0] for row in df2_labels.iterrows() if row[1].max() ==1]
        house_ar = []
        house_ar = House_ID.loc[house_index]
        return house_index, House_ID, house_ar
    elif df2_labels.shape[1]>1:
        house_index = []
        house_index = [row[0] for row in df2_labels.iterrows() if row[1].max() ==1]
        return house_index
    else:
        house_index = []
        house_index = [row for row,value in enumerate(df2_labels) if value ==1]
        return house_index

#identifying which households did not own EV's
def does_not_own_EV(df_labels):
    df2_labels = df_labels.copy()
    if 'House ID' in df2_labels:
        House_ID = df2_labels['House ID']
        df2_labels.drop('House ID', inplace = True, axis =1)
        house_index = []
        house_index = [row[0] for row in df2_labels.iterrows() if row[1].max() ==0]
        house_ar = []
        house_ar = [house for row, house in enumerate(House_ID) for idx in house_index if row == idx]
        return house_index, House_ID, house_ar
    elif df2_labels.shape[1]>1:
        house_index = []
        house_index = [row[0] for row in df2_labels.iterrows() if row[1].max() ==0]
        return house_index
    else:
        house_index = []
        house_index = [row for row,value in enumerate(df2_labels) if value ==0]
        return house_index

def boosted_tree(df, df_labels):
    df2,df2_labels = df.copy(),df_labels.copy()
    X_train, y_train = df2.iloc[:,424],df2_labels.iloc[:,424]
    house_index = owns_EV(df2_labels.iloc[:,760])
    X_test, y_test = df2.iloc[house_index,760],df2_labels.iloc[house_index,760]
    gbr.fit(X_train.values.reshape(1586,1),y_train)
    gbr_predicted = gbr.predict(X_test.values.reshape(len(house_index),1))
    return gbr, gbr_predicted, y_train, y_test


#determines the avg kwh usage for both homes that own EV & homes that do not
def home_avg(df, df_labels):
    df2,df2_labels = df.copy(),df_labels.copy()
    df2.drop('House ID', inplace = True, axis =1)
    df2_labels.drop('House ID',inplace = True, axis=1)
    owns_EV_ar = []
    does_not_own_EV_ar=[]
    house_index = owns_EV(df2_labels)
    print "ones index",house_index
    owns_EV_ar=[row[1].mean() for row in df2.loc[house_index,:].iterrows()]
    house_index = does_not_own_EV(df2_labels)
    does_not_own_EV_ar=[row[1].mean() for row in df2.loc[house_index,:].iterrows()]

    return owns_EV_ar, does_not_own_EV_ar

#creates tuple that has both kwh usage for interval and has "1" if
#an EV is plugged and a "0" if an EV is not plugged in
def EV_plugged_in(df, df_labels):
    df2,df2_labels = df.copy(),df_labels.copy()
    df2.drop('House ID', inplace = True, axis =1)
    df2_labels.drop('House ID',inplace = True, axis=1)
    kwh_ar= []
    ev_counts_per_interval = []
    dist = []
    for row_x, row_y in izip(df2.iterrows(),df2_labels.iterrows()):
        dist.append(zip(row_x[1],row_y[1]))

    return dist

def avg_func(dist):
    #2.3  ...the incremental amount on avg each house that owns an ev uses when ev is plugged in
    house_avg = []
    mu_by_house=[]
    for house in dist:
        zeroes=[]
        ones=[]
        all_hours=[]
        for val in house:
            if val[1]==1:
                ones.append(val[0])
            else:
                zeroes.append(val[0])
            all_hours.append(val[0])
        if not ones:
            house_avg.append((np.mean(zeroes),0))
        else:
            house_avg.append((np.mean(zeroes),np.mean(ones)))
        mu_by_house.append(np.mean(all_hours))
    return house_avg,mu_by_house

#creating a timeseries NeuralNet
def ts_NN_fun(df,df_labels,look_back,df_final_test):
    df2,df2_labels, df2_final_test = df.copy(),df_labels.copy(),df_final_test
    df2.drop('House ID', inplace = True, axis =1)
    df2_labels.drop('House ID',inplace = True, axis=1)
    df2_final_test.drop('House ID',inplace = True, axis=1)

    #averaging each Interval
    df_avg = []
    df_avg=[np.mean(df2[col]) for col in df2.loc[:,'Interval_1':]]

    #creating training set
    X_train= scaler.fit_transform(np.reshape(df2.loc[row,'Interval_1':'Interval_1344'].values.T.ravel(),(1344,1)))
    y_train = scaler.fit_transform(np.reshape(df2.loc[row,'Interval_2':'Interval_1345'].values.T.ravel(),(1344,1)))
    #X_train = scaler.fit_transform(df_avg[0:-1])
    #y_train = scaler.fit_transform(df_avg[1:])
    #LSTM X needs to be in [samples, time steps, features].
    X_train = np.reshape(X_train,(X_train.shape[0],look_back,1))


    #create and fit LSTM  network
    model = Sequential()
    model.add(LSTM(input_dim=1,output_dim=X_train.shape[0],return_sequences=True
    ))
    model.add(Dropout(.2))
    model.add(LSTM(100,return_sequences=False))
    model.add(Dropout(.2))
    model.add(Dense(output_dim=look_back))
    model.add(Activation("linear"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print "X_train", X_train.shape, type(X_train)
    print "y_train", y_train.shape, type(y_train)
    print model.summary()
    model.fit(X_train, y_train, nb_epoch=5, batch_size=1, verbose=2)

    return model,X_train, y_train

#creating test set
def test_data(df_final_test,row_num):
    df2_final_test = df_final_test.copy()
    df2_final_test.drop('House ID',inplace=True,axis=1)
    X_test = scaler.fit_transform(np.reshape(df2_final_test.loc[row_num,'Interval_1':'Interval_2879'].values,(2879,1)))
    y_test=scaler.fit_transform(df2_final_test.loc[row_num,'Interval_2':].values,(2879,1))
    X_test = np.reshape(X_test,(X_test.shape[0],look_back,1))
    return X_test,y_test

#using the neural net to predict
def make_predictions(model,X_train,y_train,df_final_test):
    test_score_ar = []
    pred_train_ar = []
    pred_test_ar = []
    prob_train_ar=[]
    prob_test_ar =[]
    pred_train = model.predict(X_train)
    prob_train=pred_train
    pred_train = scaler.inverse_transform(pred_train)
    y_train = scaler.inverse_transform(y_train)
    print y_train.shape
    for row_num in df_final_test.index:
        X_test,y_test = test_data(df_final_test,row_num)

        pred_test = model.predict(X_test)
        prob_test = pred_test
        prob_test_ar.append(prob_test)

        # invert predictions
        pred_test=scaler.inverse_transform(pred_test)
        pred_test_ar.append(pred_test)
        y_test = scaler.inverse_transform(y_test)
        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(y_train, pred_train))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(y_test, pred_test))
        print('Test Score: %.2f RMSE' % (testScore))
        print "row #", row_num
        test_score_ar.append(testScore)
    prob_train_ar=np.array(prob_train_ar)
    prob_test_ar=np.array(prob_test_ar)
    pred_test_ar= np.array(pred_test_ar)
    return pred_train,pred_test, prob_train, prob_test,pred_test_ar, prob_test_ar,test_score_ar


#plotting the results of the neural net
def plotting(df, df_final_test, pred_train,pred_test,look_back):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df2,df2_final_test = df.copy(),df_labels.copy()
    df2.drop('House ID', inplace = True, axis =1)
    df_avg = []
    df_avg=[np.mean(df2[col]) for col in df2.loc[:,'Interval_1':]]
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(1,1,1)
    col_range = (np.arange(1,2880))
    print "col_range", len(col_range)
    kwh_train_range= df_avg[1:]
    kwh_test_range = df2_final_test.iloc[0,col_range]
    kwh_range= np.concatenate((kwh_train_range,kwh_test_range))
    ax.plot(kwh_range,color='k',label='Actual kwh usage')
    ax.plot(pred_train,color='b',label='Predictions on train set')

    testPredictPlot = np.append(np.full(len(pred_train),np.nan),pred_test)
    ax.plot(testPredictPlot,color='r',label='Predictions on test set')
    plt.ylabel('kwh usage')
    plt.savefig('kwh usagev8')
    pass

#calculating recall
def recall(df_labels,houses,interval_range,prob_train,prob_test):
    df2_labels= df_labels.copy()
    df2_labels.drop('House ID',inplace = True, axis=1)
    cols=df2_labels.loc[houses,interval_range][df2_labels.loc[houses,interval_range]==1].index
    cols = [int(unidecode(x).split('_')[1]) for x in cols]
    train_recall = 0
    for val in prob_train[cols]:
        #using current max probability form NN... replace when higher probability is achieved
        if val >(np.max(prob_train)/2):
            train_recall+=1

    #cols=df2_labels.loc[houses,interval_range][df2_labels.loc[houses,interval_range]==1].index
    return train_recall


#finding rows that contain null data
def find_NaN(df):
    null_row_ar=[]
    for row in df.index:
        if np.isnan(df.iloc[row,:].values).any():
            null_row_ar.append(row)

    return null_row_ar

if __name__ == "__main__":

    np.random.seed(42)
    df, df_labels,df_final_test = get_data()
    #remove NaN's
    null_row_ar = find_NaN(df)
    df.drop(null_row_ar,axis =0, inplace=True)
    df_labels.drop(null_row_ar,axis =0, inplace=True)
    null_row_ar = find_NaN(df_final_test)
    df_final_test.drop(null_row_ar,axis=0,inplace=True)
    house_index, House_ID, house_owns_ev = owns_EV(df_labels)
    df_charge=time_of_charge(df,df_labels)

    dist,row_x,row_y = EV_plugged_in(df,df_labels)
    house_avg,mu_by_house = avg_func(dist)

    owns_EV_ar, does_not_own_EV_ar=home_avg(df,df_labels)
    ts_log = lincoln_log(df,df_labels)
    look_back=1
    scaler = MinMaxScaler(feature_range=(0, 1))
    model,X_train, X_test, y_train, y_test = ts_NN_fun(df,df_labels,1,df_final_test)
    pred_train,pred_test, prob_train, prob_test,pred_test_ar, prob_test_ar,test_score_ar=make_predictions(model,X_train, y_train,df)
    plotting(df,df_final_test ,pred_train,pred_test,look_back)
