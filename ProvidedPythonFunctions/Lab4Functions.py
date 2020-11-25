#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:16:05 2020

@author: patrickmayerhofer

These functions are specifically for BPK 409 - Wearable Technology and Human Physiology
"""

"""import libraries"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix
import seaborn as sns
import glob, os
from tensorflow import keras



"""This function helps graphically labelling the data 
and saves the labelled file as LabelledData.csv in a new
folder called LabelledData in the working directory.
Input: file name of the activity file"""
def LabelData(file_name):
    def get_activities(x,y,z):
        def zoom_ginput(x,y,z):
            def tellme(s):
                print(s)
                plt.title(s, fontsize=16)
                plt.draw()
                
            plt.clf()
            plt.setp(plt.gca(), autoscale_on=True)
            plt.plot(x)
            plt.plot(y)
            plt.plot(z)
           
            tellme('Click once to start zoom')
            plt.waitforbuttonpress()
            
            while True:
                tellme('Select two corners of zoom, enter button to finish')
                pts = plt.ginput(2, timeout=-1)
                if len(pts) < 2:
                    break
                (x0, y0), (x1, y1) = pts
                xmin, xmax = sorted([x0, x1])
                ymin, ymax = sorted([y0, y1])
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)
              
                
            tellme('Choose start of activity')    
            s = plt.ginput(1)
            tellme('Choose end of activity, then go back and type in activity')   
            e = plt.ginput(1)
            tellme('Go back and type in activity')  
            s1 = s[0]
            e1 = e[0]
            start = s1[0].astype(np.int64)
            end = e1[0].astype(np.int64)
            plt.show()
            return start,end
            
        n_activities = int(input("How often did you do activities inbetween standing: "))
        start = np.empty(n_activities)
        end = np.empty(n_activities)
        activities = []
        
        for i in range(n_activities):
            start[i],end[i] = zoom_ginput(x,y,z)
            a = input("Activity of this part: ")
            activities.append(a)
        
        return start, end, activities
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    
    """import data"""
    column_names = [
      'x',
      'y',
      'z',
      't',  
    ]
    """Path"""
    root_path = os.path.realpath(os.path.join(os.getcwd()))
    save_path = root_path + '/LabelledData/'
    
    """Import"""
    df = pd.read_csv(file_name, sep="\t", header=None, names=column_names)
    
    
    """label data"""
    # mark beginning, end and activity name
    start, end, activities = get_activities(df.x, df.y, df.z)
    start = start.astype(np.int64)
    end = end.astype(np.int64)
    # add activity to df
    df_activity = [0 for x in range(len(df))]
    for i in range(len(activities)):
        for u in range(start[i], end[i]):
            df_activity[u] = activities[i]
    
    a = [i for i, e in enumerate(df_activity) if e == 0] 
    for i in a:
        df_activity[i] = "Standing"
    df['activity'] = df_activity
    
    # plot with two axes
    fig,ax=plt.subplots()
    ax.plot(df.t, df.x)
    ax.plot(df.t, df.y)
    ax.plot(df.t, df.z)
    ax.set_ylabel("Accelerations (G)")
    ax.set_xlabel("Time (ms)")
    
    ax2=ax.twinx()
    ax2.plot(df.t, df.activity, color = 'black')
    

    #plt.plot(df.t[df.activity != 'Standing'], df.x[df.activity != 'Standing'], color = 'green')
    
    # Create target Directory if don't exist and save data
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("Directory " , save_path ,  " Created ")
    df.to_csv(save_path + '/LabelledData.csv')
    
 
"""Splits the full dataset into windows.
Input: X, y, length of window, sliding length"""
def create_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)


"""Puts all the data into one file. 
Will load every LabelledDataN.csv file in the folder LabelledData
where N is the ID of the subject or dataset
Output: a datafile with the data of all subjects together
"""
def import_data():
    column_names_df = [
      'x',
      'y',
      'z',
      't',
      'activity'
    ]
    
    column_names_alldata = [
      'x',
      'y',
      'z',
      't',
      'activity', 
      'id'
    ]
    
    """Path"""
    root_path = os.path.realpath(os.path.join(os.getcwd()))
    load_path = root_path + '/LabelledData/'
    
    
    # load one file, delete first column, add an 'id' column, and append to the big dataframe
    num_files = len(glob.glob1(load_path,"*.csv"))
    all_data = pd.DataFrame(columns = column_names_alldata)
    os.chdir(load_path)
    for i in range(num_files):
        file = 'LabelledData'+ str(i+1) + '.csv'
        df = pd.read_csv(load_path + file, names = column_names_df, skiprows= 25, skipfooter = 25, engine = 'python')
        #df = df.drop(df.columns[0], axis=1) # delete first column
        sub_id = np.empty(len(df))
        sub_id[:] = i + 1
        df['id'] = sub_id
        all_data = all_data.append(df)
        print('File loaded: ' + file)
    all_data = all_data.dropna()
    all_data = all_data.reset_index(drop = True)     
    return all_data

"""Plot a confusion matrix
Input: y_true, y_predicted, class_names"""        
def plot_cm(y_true, y_pred, class_names):
      cm = confusion_matrix(y_true, y_pred)
      fig, ax = plt.subplots(figsize=(18, 16)) 
      ax = sns.heatmap(
          cm, 
          annot=True, 
          fmt="d", 
          cmap=sns.diverging_palette(220, 20, n=7),
          ax=ax
      )
    
      plt.ylabel('Actual')
      plt.xlabel('Predicted')
      ax.set_xticklabels(class_names)
      ax.set_yticklabels(class_names)
      b, t = plt.ylim() # discover the values for bottom and top
      b += 0.5 # Add 0.5 to the bottom
      t -= 0.5 # Subtract 0.5 from the top
      plt.ylim(b, t) # update the ylim(bottom, top) values
      plt.show() # ta-da!
   
"""Create the bidirectional lstm model. Needs X_train and 
y_train to understand the nature of the model input and output.
Input: X_train, ytrain 
Output: The prepared model"""
def create_model(X_train, y_train):
    model = keras.Sequential()
    model.add(
        keras.layers.Bidirectional(
          keras.layers.LSTM(
              units=128,
              input_shape=[X_train.shape[1], X_train.shape[2]]
          )
        )
    )
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
  
    return model
    
