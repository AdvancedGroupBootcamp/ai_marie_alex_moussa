# import lib_2

import matplotlib.pyplot as plt
import seaborn as sns
import os
from IPython.display import display

# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler



def eda_base(df, correlation = 0, n_round = 2):
    display(df.head())
    display(df.info())
    print('Shape', df.shape)
    display(df.describe().round(n_round))
    print('isNa?', df.isna().any().sum())
    print('Duplicates?', df.duplicated().sum())
    
    
def plot_corr(df, title = '', x_rotation = 45):
    # correlation - by Heatmap the relationship between the features.
    plt.figure(figsize=(10,6))
    # sns.heatmap(df.corr(), cmap=plt.cm.Reds, annot=True, fmt='.3g')
    sns.heatmap(df.corr(), cmap='Blues', annot=True, fmt='.3g')
    plt.yticks(rotation = 0) 
    plt.xticks(rotation = x_rotation) 

    plt.title(title, fontsize=13)
    plt.show()

    
def plot_hist(df):
	# distributions of the variables/features.
	df.hist(figsize=(12,8),bins=20)
	plt.show()

    
def read_all_files_from_dir(path = './'):
    files = [file for file in os.listdir(path)]
#     for file in files:
#         print (file)
    return files


def set_fstline_as_header(df):
    new_header = df.iloc[0] #grab the first row for the header
    df = df[1:] #take the data less the header row
    df.columns = new_header 
    return df



def get_scaled_data(df, scaler = StandardScaler()):
    scaler.fit(df)
    scaled_data = scaler.transform(df)
    return scaled_data