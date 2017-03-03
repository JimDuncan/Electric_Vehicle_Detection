import pandas as pd
import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
import scipy.stats as scs
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score,precision_score
from sklearn.metrics import precision_score,accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from itertools import izip
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from imblearn.over_sampling import   SMOTE
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler


#
sm=SMOTE(k=5,random_state=42,ratio=(1.0/3))
svm = SVC(kernel='poly',class_weight={1:9})
gbr = GradientBoostingRegressor(random_state=42)



def get_data():
    df = pd.read_csv('EV_files/EV_train.csv')
    df_labels = pd.read_csv('EV_files/EV_train_labels.csv')
    df_final_test= pd.read_csv('EV_files/EV_test.csv')
    return df,df_labels,df_final_test

def owns_EV(df_labels):
    df2_labels = df_labels.copy()
#    Interval_cols = df2_labels.columns.tolist()[1:]
    if 'House ID' in df2_labels:
        House_ID = df2_labels['House ID']
        df2_labels.drop('House ID', inplace = True, axis =1)
        index_ar = []
        index_ar = [row[0] for row in df2_labels.iterrows() if row[1].max() ==1]
        house_ar = []
        house_ar = [house for row, house in enumerate(House_ID) for idx in index_ar if row == idx]
        return index_ar, House_ID, house_ar
    elif df2_labels.shape[1]>1:
        index_ar = []
        index_ar = [row[0] for row in df2_labels.iterrows() if row[1].max() ==1]
        return index_ar
    else:
        index_ar = []
        index_ar = [row for row,value in enumerate(df2_labels) if value ==1]
        return index_ar

def does_not_own_EV(df_labels):
    df2_labels = df_labels.copy()
#    Interval_cols = df2_labels.columns.tolist()[1:]
    if 'House ID' in df2_labels:
        House_ID = df2_labels['House ID']
        df2_labels.drop('House ID', inplace = True, axis =1)
        index_ar = []
        index_ar = [row[0] for row in df2_labels.iterrows() if row[1].max() ==0]
        house_ar = []
        house_ar = [house for row, house in enumerate(House_ID) for idx in index_ar if row == idx]
        return index_ar, House_ID, house_ar
    elif df2_labels.shape[1]>1:
        index_ar = []
        index_ar = [row[0] for row in df2_labels.iterrows() if row[1].max() ==0]
        return index_ar
    else:
        index_ar = []
        index_ar = [row for row,value in enumerate(df2_labels) if value ==0]
        return index_ar


def time_of_charge(df,df_labels):
    df2,df2_labels = df.copy(),df_labels.copy()
    df2.drop('House ID', inplace = True, axis =1)
    df2_labels.drop('House ID',inplace = True, axis=1)
    df_charge = df2.multiply(df2_labels)
    return df_charge

def binom_df(df_labels,index_ar):
    df_binom = df_labels.copy()
    df_binom = df_binom.iloc[index_ar,:]
    #Within the set of Houses that own an EV, count of how many of those houses are charging their EV on a given hour
    binom_ar = []
    binom_percent_ar = []
    for col in df_binom:
        binom_ar.append(df_binom[col].sum())
    binom_ar.pop(0)
    binom_percent_ar =[1.00*count/len(index_ar) for count in binom_ar]
    return df_binom, binom_ar, binom_percent_ar

def ev_compare(df_labels):
    df2_labels = df_labels.copy()
    df2_labels.drop('House ID',inplace=True,axis=1)
    y_train = df2_labels.loc[:,'Interval_1':'Interval_1344']
    y_test =df2_labels.loc[:,'Interval_1345':'Interval_2688']
    month_by_month = []
    for col_train,col_test in izip(y_train,y_test):
        month_by_month.append(y_test[col_test].sum()-y_train[col_train].sum())
    y_train = df2_labels.loc[:,'Interval_1':'Interval_336']
    y_test_1 = df2_labels.iloc[:,336:(y_train.shape[1]+336)]
    y_test_2 = df2_labels.iloc[:,y_test_1.shape[1]*2:(y_test_1.shape[1]*2+336)]
    y_test_3 = df2_labels.iloc[:,y_test_1.shape[1]*3:(y_test_1.shape[1]*3+336)]
    y_test_4 = df2_labels.iloc[:,y_test_1.shape[1]*4:(y_test_1.shape[1]*4+336)]
    y_test_5 = df2_labels.iloc[:,y_test_1.shape[1]*5:(y_test_1.shape[1]*5+336)]
    y_test_6 = df2_labels.iloc[:,y_test_1.shape[1]*6:(y_test_1.shape[1]*6+336)]
    y_test_7 = df2_labels.iloc[:,y_test_1.shape[1]*7:(y_test_1.shape[1]*7+336)]
    week_by_week=[y_test_1.sum().values-y_train.sum().values]
    week_by_week.append(y_test_2.sum().values-y_test_1.sum().values)
    week_by_week.append(y_test_3.sum().values-y_test_2.sum().values)
    week_by_week.append(y_test_4.sum().values-y_test_3.sum().values)
    week_by_week.append(y_test_5.sum().values-y_test_4.sum().values)
    week_by_week.append(y_test_6.sum().values-y_test_5.sum().values)
    week_by_week.append(y_test_7.sum().values-y_test_6.sum().values)


    return month_by_month,week_by_week



def random_forest_model(df, df_labels,index_ar,house_avg,mu_by_house):
    rf = RandomForestClassifier(criterion='gini',n_estimators=20,max_features =1,random_state=42)
    gbc = GradientBoostingClassifier(learning_rate=0.01,min_samples_split=15,min_samples_leaf=25,max_depth=5,max_features='sqrt',random_state=42)

    df2,df2_labels = df.copy(),df_labels.copy()
    df2.drop('House ID',inplace=True,axis=1)
    df2_labels.drop('House ID',inplace=True,axis=1)
    non_active=[house for house in df.index.tolist() if house not in index_ar]
    active_houses=df2.index.tolist()
#    X_train, y_train = df2.loc[:,'Interval_337':'Interval_672'],df2_labels.loc[:,'Interval_337':'Interval_672']
#    X_test, y_test = df2.loc[:,'Interval_673':'Interval_1008'],df2_labels.loc[:,'Interval_673':'Interval_1008']
    X_train, y_train = df2.loc[:,'Interval_1':'Interval_1344'],df2_labels.loc[:,'Interval_1':'Interval_1344']
    X_test, y_test = df2.loc[:,'Interval_1345':'Interval_2688'],df2_labels.loc[:,'Interval_1345':'Interval_2688']
    print "shape",X_test.shape, X_train.shape
    og_coltrain_list = X_train.columns.tolist()
    insert_col_index=1
    for column_x, column_y in izip(X_train,y_train):
        new_col=[]
        for cnt in active_houses:
            new_col.append(mu_by_house[active_houses.index(cnt)]-X_train[column_x][cnt])
            # if y_train[column_y][cnt]==1:
            #     new_col.append(house_avg[active_houses.index(cnt)][1]-X_train[column_x][cnt])
            # else:
            #     new_col.append(house_avg[active_houses.index(cnt)][0]-X_train[column_x][cnt])
        X_train.insert(insert_col_index,"x-" + column_x,new_col)
        insert_col_index+=2
    insert_col_index=1
    for column_x, column_y in izip(X_test,y_train):
        new_col=[]
        for cnt in active_houses:
            new_col.append(mu_by_house[active_houses.index(cnt)]-X_test[column_x][cnt])
            # if y_train[column_y][cnt]==1:
            #     new_col.append(house_avg[active_houses.index(cnt)][1]-X_test[column_x][cnt])
            # else:
            #     new_col.append(house_avg[active_houses.index(cnt)][0]-X_test[column_x][cnt])
        X_test.insert(insert_col_index,"x-" + column_x,new_col)
        insert_col_index+=2
    accuracy_ar = []
    print "shape",X_test.shape, X_train.shape
    # cnt =0
    # for col_train, col_test in izip(X_train, X_test):
    #     rf.fit(X_train[col_train].values.reshape(X_train.shape[0],1),y_train[col_train].values)
    #     rf_predicted = rf.predict(X_test[col_test].values.reshape(X_test.shape[0],1))
    #     accuracy_ar.append((y_train[col_train].sum(),rf_predicted.sum()))
    #     cnt +=1
    #     #print cnt

    cnt =0
    precision=[]
    recall=[]
    acc=[]
    for col in og_coltrain_list:
        col_num = int(col.split('_')[1])-1
        print "cnt",cnt
        end_col = "x-"+col
        print "col", col_num
        #col_list = [col_num*2,(col_num*2)+1,(col_num*2)+672,(col_num*2)+672+1,(col_num*2)+672*2,(col_num*2)+672*2+1,(col_num*2)+672*3,(col_num*2)+672*3+1]
        #print col_list
        if y_train.iloc[:,col_num].sum()<=6:
            rando=np.random.randint(1586,size=6)
            y_train.loc[rando,col]=1
        X_res,y_res=sm.fit_sample(X_train.ix[:,[col_num,col_num+1]].values,y_train.loc[:,col].values)
        gbc.fit(X_res,y_res)
        #gbc.fit(X_train.ix[index_ar,[col_num,col_num+1]].values,y_train.loc[index_ar,col].values)
        #test_col = [col_num,col_num+1,col_num,col_num+1,col_num,col_num+1,col_num,col_num+1]
        gbc_predicted=gbc.predict(X_test.ix[:,[col_num,col_num+1]].values)
        #probs=rf.predict_proba(X_test.ix[:,[col_num,col_num+1]].values)
        #accuracy_ar.append((y_test.iloc[:,col_num].sum(),rf_predicted.sum(),probs))
        precision.append(precision_score(y_test.iloc[:,col_num],gbc_predicted))
        recall.append(recall_score(y_test.iloc[:,col_num],gbc_predicted))
        acc.append((y_test.iloc[:,col_num].sum(),gbc_predicted.sum()))

    return X_train,X_test, accuracy_ar, y_train, y_test,precision,recall,acc

def random_forest_model_cont(df, df_labels,index_ar,house_avg):
    rf = RandomForestRegressor(random_state=42)
    df2,df2_labels = df.copy(),df_labels.copy()
    df2.drop('House ID',inplace=True,axis=1)
    df2_labels.drop('House ID',inplace=True,axis=1)
    active_houses=df2.index.tolist()

    index_ar.pop(len(index_ar)-1)
    X_train, y_train = df2.loc[:,'Interval_1':'Interval_1344'],df2_labels.loc[:,'Interval_1':'Interval_1344']
    X_test, y_test = df2.loc[:,'Interval_1345':'Interval_2688'],df2_labels.loc[:,'Interval_1345':'Interval_2688']
    print "shape",X_test.shape, X_train.shape
    og_coltrain_list = X_train.columns.tolist()
    insert_col_index=0
    y_train_cont=pd.DataFrame()
    for column_x, column_y in izip(X_train,y_train):
        new_col=[]
        for cnt in active_houses:
            if y_train[column_y][cnt]==1:
                new_col.append(house_avg[active_houses.index(cnt)][1]-X_train[column_x][cnt])

            else:
                # print "house_avg", house_avg[cnt][0]
                # print "X_train", X_train[column_x][cnt]
                new_col.append(house_avg[active_houses.index(cnt)][0]-X_train[column_x][cnt])

        y_train_cont.append(new_col)
        y_train_cont.insert(insert_col_index,"x-" + column_x,new_col)
        insert_col_index+=1
    insert_col_index=0
    y_test_cont=pd.DataFrame()
    for column_x, column_y in izip(X_test,y_train):
        new_col=[]
        for cnt in active_houses:
            if y_train[column_y][cnt]==1:
                new_col.append(house_avg[active_houses.index(cnt)][1]-X_test[column_x][cnt])

            else:
                new_col.append(house_avg[active_houses.index(cnt)][0]-X_test[column_x][cnt])


        y_test_cont.append(new_col)
        y_test_cont.insert(insert_col_index,"x-" + column_x,new_col)
        insert_col_index+=1

    mse_gbr = []
    cvs_gbr=[]
    mse_rf=[]
    cvs_rf=[]


    cnt =0
    for col in og_coltrain_list:
        col_num = int(col.split('_')[1])-1
        col_list = "x-"+col
        rf.fit(X_train.loc[:,col].values.reshape(1586,1),y_train.loc[:,col].values)
        gbr.fit(X_train.loc[:,col].values.reshape(1586,1),y_train.loc[:,col].values)
        gbr_predicted=gbr.predict(X_test.iloc[:,col_num].values.reshape(1586,1))
        rf_predicted=rf.predict(X_test.iloc[:,col_num].values.reshape(1586,1))
        #accuracy_ar.append((y_test_cont.iloc[:,col_num].values,gbr_predicted))

        mse = mean_squared_error(gbr_predicted,y_test.iloc[:,col_num])
        mse_gbr.append(np.mean(mse))
        mse = mean_squared_error(rf_predicted,y_test.iloc[:,col_num])
        mse_rf.append(np.mean(mse))
        # cvs=-1.0*cross_val_score(gbr,X_train.loc[index_ar,col].values.reshape(483,1),y_train_cont.loc[index_ar,col_list].values,scoring='neg_mean_squared_error',cv=
        # 10)
        # cvs_gbr.append(np.mean(cvs))
        # cvs=-1.0*cross_val_score(rf,X_train.loc[index_ar,col].values.reshape(483,1),y_train_cont.loc[index_ar,col_list].values,scoring='neg_mean_squared_error',cv=
        # 10)
        # cvs_rf.append(np.mean(cvs))

#    print accuracy_ar[0:10]
    return X_train,X_test, y_train_cont, y_test_cont,mse_gbr,mse_rf, cvs_gbr,cvs_rf


def boosted_tree(df, df_labels):
    df2,df2_labels = df.copy(),df_labels.copy()
    X_train, y_train = df2.iloc[:,424],df2_labels.iloc[:,424]
    index_ar = owns_EV(df2_labels.iloc[:,760])
    X_test, y_test = df2.iloc[index_ar,760],df2_labels.iloc[index_ar,760]
    gbr.fit(X_train.values.reshape(1586,1),y_train)
    gbr_predicted = gbr.predict(X_test.values.reshape(len(index_ar),1))
    return gbr, gbr_predicted, y_train, y_test

def k_means_model(df, df_labels):
    df2,df2_labels = df.copy(),df_labels.copy()
    owns_cars = df2_labels.iloc[:,424].sum()
    kmeans= KMeans(n_clusters =owns_cars)
    index_zero = does_not_own_EV(df2_labels.iloc[:,424])
    X_train,y_train=df2.iloc[index_zero,424],df2_labels.iloc[index_zero,424]
    print X_train.values
    kmeans.fit(X_train.values)
    return kmeans



def support_vector(df, df_labels):
    df2,df2_labels = df.copy(),df_labels.copy()
    X_train, y_train = df2.iloc[:,424],df2_labels.iloc[:,424]
    X_test, y_test = df2.iloc[:,760],df2_labels.iloc[:,760]
    svm.fit(X_train.values.reshape(1589,1),y_train)
    svm_predicted = svm.predict(X_test.values.reshape(1589,1))
    return svm, svm_predicted, y_train, y_test

def logistic_regress(df, df_labels,index_ar):
    df2,df2_labels = df.copy(),df_labels.copy()
    df2.drop('House ID',inplace=True,axis=1)
    df2_labels.drop('House ID',inplace=True,axis=1)
    X_train, y_train = df2.loc[:,'Interval_1':'Interval_1344'],df2_labels.loc[:,'Interval_1':'Interval_1344']
    X_test, y_test = df2.loc[:,'Interval_1345':'Interval_2688'],df2_labels.loc[:,'Interval_1345':'Interval_2688']
    og_col_list = X_train.columns.tolist()
    lr = LogisticRegression(penalty='l1')
    accuracy_ar=[]
    for col in og_col_list:
        col_num = int(col.split('_')[1])-1
        lr.fit(X_train.loc[index_ar,col].values.reshape(484,1),y_train.loc[index_ar,col].values)
        lr_predicted = lr.predict(X_test.iloc[:,col_num].values.reshape(1586,1))
        probs = lr.predict_proba(X_test.iloc[:,col_num].values.reshape(1586,1))
        accuracy_ar.append((y_test.iloc[:,col_num].sum(),lr_predicted.sum(),probs))

    return lr, lr_predicted, y_train, y_test, accuracy_ar

def home_avg(df, df_labels):
    df2,df2_labels = df.copy(),df_labels.copy()
    df2.drop('House ID', inplace = True, axis =1)
    df2_labels.drop('House ID',inplace = True, axis=1)
    owns_EV_ar = []
    does_not_own_EV_ar=[]
    index_ar = owns_EV(df2_labels)
    print "ones index",index_ar
    owns_EV_ar=[row[1].mean() for row in df2.loc[index_ar,:].iterrows()]
    index_ar = does_not_own_EV(df2_labels)
    does_not_own_EV_ar=[row[1].mean() for row in df2.loc[index_ar,:].iterrows()]

    return owns_EV_ar, does_not_own_EV_ar




def totals(df, df_labels):
    df2,df2_labels = df.copy(),df_labels.copy()
    df2.drop('House ID', inplace = True, axis =1)
    df2_labels.drop('House ID',inplace = True, axis=1)
    kwh_ar= []
    ev_counts_per_interval = []
    # kwh_ar = df2.values().ravel()
    # ev_counts_per_interval = df2_labels.values.ravel()
    # for col_x, col_y in izip(df2,df2_labels):
    #     kwh_ar.append(df2[col_x].sum())
    #     ev_counts_per_interval.append(df2_labels[col_y].sum())
    dist = []
    print "shapes", df2.shape
    print "shapes_labels", df2_labels.shape

    for row_x, row_y in izip(df2.iterrows(),df2_labels.iterrows()):
        dist.append(zip(row_x[1],row_y[1]))

    #
    # dist = sorted(dist, key=lambda x:x[1])
    # for val in dist:
    #     kwh_ar.append(val[0])
    #     ev_counts_per_interval.append(val[1])
    #
    # zeroes = []
    # ones = []
    # for values in dist:
    #     if values[1]==1:
    #         ones.append(values[0])
    #     else:
    #         zeroes.append(values[0])

    return dist,row_x,row_y
    #ones, zeroes, kwh_ar, ev_counts_per_interval
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

def get_NN_model(X_train,y_train, dropout = 50, activation = 'softmax', node_count = 64, init_func='uniform', layers = 1):
    #
    # model.add(Dense(node_count, init=init_func, input_dim=X.shape[1]))
	# model.add(Activation(activation))
	# model.add(Dropout(dropout / 100.0))
    #
    # for layer in range(layers):
    #     model.add(Dense(node_count,init=init_func))
    #     model.add(Activation(activation))
    #     model.add(Dropout(dropout/100.0))
    #
    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	# model.compile(loss='mse', optimizer=sgd)

    #
    # model = Sequential()
    # model.add(Dense(input_dim=X_train.shape[1],
    #                  output_dim=200,
    #                  init='uniform',
    #                  activation='relu'))
    # model.add(Dense(input_dim=200,
    #                  output_dim=200,
    #                  init='uniform',
    #                  activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(input_dim=200,
    #                  output_dim=200,
    #                  init='uniform',
    #                  activation='relu'))
    # model.add(Dense(input_dim=200,
    #                  output_dim=y_train.shape[1],
    #                  init='uniform',
    #                  activation='softmax'))

    model = Sequential()
    model.add(LSTM(4,input_dim=1))
    model.add(Dense(1))


    sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["recall"] )

    return model

def ts_NN_fun(df,df_labels,look_back,df_final_test):
    df2,df2_labels, df2_final_test = df.copy(),df_labels.copy(),df_final_test
    df2.drop('House ID', inplace = True, axis =1)
    df2_labels.drop('House ID',inplace = True, axis=1)
    df2_final_test.drop('House ID',inplace = True, axis=1)

    #averaging each Interval
    df_avg = []
    df_avg=[np.mean(df2[col]) for col in df2.loc[:,'Interval_1':]]

    #creating training set
    #X_train= scaler.fit_transform(np.reshape(df2.loc[28:29,'Interval_1':'Interval_1344'].values.T.ravel(),(2688,1)))
    #y_train = scaler.fit_transform(np.reshape(df2.loc[28:29,'Interval_2':'Interval_1345'].values.T.ravel(),(2688,1)))
    X_train = scaler.fit_transform(df_avg[0:-1])
    y_train = scaler.fit_transform(df_avg[1:])
    #LSTM X needs to be in [samples, time steps, features].
    X_train = np.reshape(X_train,(X_train.shape[0],look_back,1))


    #create and fit LSTM  network
    # model = Sequential()
    # model.add(LSTM(4, input_dim=look_back))
    # model.add(Dense(1))
    # model.compile(loss='mean_squared_error', optimizer='adam')
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

    return model,X_train, X_test, y_train, y_test

def test_data(df_final_test,row_num):
    df2_final_test = df_final_test.copy()
    df2_final_test.drop('House ID',inplace=True,axis=1)
    X_test = scaler.fit_transform(np.reshape(df2_final_test.iloc[row_num,0:-1].values,(2879,1)))
    y_test=scaler.fit_transform(df2_final_test.iloc[row_num,1:].values,(2879,1))
    X_test = np.reshape(X_test,(X_test.shape[0],look_back,1))
    return X_test,y_test

def make_predictions(model,X_train,y_train,df_final_test):
    test_score_ar = []
    pred_test_ar = []
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
    prob_test_ar=np.array(prob_test_ar)
    pred_test_ar= np.array(pred_test_ar)
    return pred_train,pred_test, prob_train, prob_test,pred_test_ar, prob_test_ar,test_score_ar

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

#
    pass



def lincoln_log(df, df_labels):
    df2,df2_labels = df.copy(),df_labels.copy()
    df2.drop('House ID', inplace = True, axis =1)
    df2_labels.drop('House ID',inplace = True, axis=1)
    ts_log = np.log(df2.loc[4,'Interval_1':'Interval_1344'].values)
    return ts_log

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





def ARIMA():
    pass

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
    index_ar, House_ID, house_ar = owns_EV(df_labels)
    df_charge=time_of_charge(df,df_labels)
#    df_binom, binom_ar, binom_percent_ar = binom_df(df_labels,index_ar)
#
    #svm, svm_predicted, y_train, y_test = support_vector(df, df_labels)

    dist,row_x,row_y = totals(df,df_labels)
    house_avg,mu_by_house = avg_func(dist)


    #month_by_month,week_by_week=ev_compare(df_labels)
    #X_train,X_test, y_train_cont, y_test_cont,mse_gbr,mse_rf, cvs_gbr,cvs_rf = random_forest_model_cont(df, df_labels,index_ar,house_avg)
    #X_train,X_test, accuracy_ar, y_train, y_test,precision,recall,acc =random_forest_model(df, df_labels,index_ar,house_avg,mu_by_house)
#    gbr, gbr_predicted, y_train, y_test = boosted_tree(df, df_labels)
    #lr, lr_predicted, y_train, y_test, accuracy_ar = logistic_regress(df, df_labels,index_ar)
    owns_EV_ar, does_not_own_EV_ar=home_avg(df,df_labels)
    ts_log = lincoln_log(df,df_labels)
    look_back=1
    scaler = MinMaxScaler(feature_range=(0, 1))
    model,X_train, X_test, y_train, y_test = ts_NN_fun(df,df_labels,1,df_final_test)
    pred_train,pred_test, prob_train, prob_test,pred_test_ar, prob_test_ar,test_score_ar=make_predictions(model,X_train, y_train,df_final_test)
    plotting(df,df_final_test ,pred_train,pred_test,look_back)

    #kmeans = k_means_model(df, df_labels)
