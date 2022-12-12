import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import metrics
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.impute import KNNImputer
from warnings import filterwarnings

filterwarnings("ignore")




from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def loadData():
    X_train=pd.read_table("ada_train.data",delim_whitespace=True,header=None)
    X_valid=pd.read_table("ada_valid.data",delim_whitespace=True,header=None)
    X_test=pd.read_table("ada_test.data",delim_whitespace=True,header=None)


    y_train=pd.read_table("ada_train.labels",delim_whitespace=True,header=None)
    y_valid=pd.read_table("ada_valid.labels",delim_whitespace=True,header=None)


    Column_train={"X"+str(i):X_train.iloc[:,i-1] for i in range(1,X_train.shape[1]+1)}
    Column_valid={"X"+str(i):X_valid.iloc[:,i-1] for i in range(1,X_valid.shape[1]+1)}
    Column_test={"X"+str(i):X_test.iloc[:,i-1] for i in range(1,X_test.shape[1]+1)}
    ada_train_data=pd.DataFrame(Column_train)
    ada_valid_data=pd.DataFrame(Column_valid)
    ada_test_data=pd.DataFrame(Column_test)
    ada_train_label=pd.DataFrame({"target":y_train.iloc[:,0]})
    ada_valid_label=pd.DataFrame({"target":y_valid.iloc[:,0]})
    ada_train_data.shape,ada_valid_data.shape,ada_test_data.shape
    ada_train_label.shape,ada_valid_label.shape
    return (ada_train_data, ada_train_label, ada_valid_data, ada_valid_label, ada_test_data)

def detectOutLayers(df):
    info = df.copy()
    model=IsolationForest(max_samples='auto', contamination=float(0.1))
    model.fit(df)
    info = info.assign(scores=model.decision_function(df))
    info = info.assign(anomaly=model.predict(df))
    anomaly=info.loc[info['anomaly']==-1]
    anomaly_index=list(anomaly.index)
    # print(anomaly_index)
    return anomaly_index



def get_percentage_of_outlier(df):
    numerical_cols=df.columns[df.nunique()>2]
    n=df.shape[0]
    outlier_size={}
    for col in numerical_cols:
        q1=df[col].quantile(0.25)
        q3=df[col].quantile(0.75)
        IQR=q3-q1
        outliers_shape=df[(df[col]<q1-1.5*IQR)].shape[0] + df[(df[col]>q3+1.5*IQR)].shape[0]
        outlier_size[col]=outliers_shape
    return outlier_size

def transformData(df):
    
    colToFix = {}
    colToTreat= []
    for col in df.columns:
        if len(df[col].unique()) > 1 :
            colToTreat.append(col)

        # df = (df-df.mean())/df.std()
        for col in colToTreat:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def normalize_num_cols(df):
    df_standard=df.copy()
    standardizing=StandardScaler()
    numerical_cols=df.columns[df.nunique()>2]
    standar_col_num=standardizing.fit_transform(df[numerical_cols])
    df_standard[numerical_cols]=standar_col_num
    return df_standard

def get_accuray(train_set,train_label, valid_set, valid_label,cv=50):
    Classifiers=[KNeighborsClassifier,LogisticRegression,DecisionTreeClassifier,
            GaussianNB,LinearDiscriminantAnalysis]
    col=["Name","Accuracy_mean","BER_mean","ROC_mean","Accuracy_std","BER_std","ROC_std"]
    df=pd.DataFrame(columns=col)
    for classifier in Classifiers:
        Name=classifier.__name__
        model=classifier().fit(train_set,train_label)
        y_pred=model.predict(valid_set)
        acc=np.mean(cross_val_score(model,train_set,train_label,scoring="accuracy",cv=cv))
        BER=np.mean(cross_val_score(model,train_set,train_label,scoring="balanced_accuracy",cv=cv))
        ROC=np.mean(cross_val_score(model,train_set,train_label,scoring="roc_auc",cv=cv))

        acc_std=np.std(cross_val_score(model,train_set,train_label,scoring="accuracy",cv=cv))
        BER_std=np.std(cross_val_score(model,train_set,train_label,scoring="balanced_accuracy",cv=cv))
        ROC_std=np.std(cross_val_score(model,train_set,train_label,scoring="roc_auc",cv=cv))

        df1=pd.DataFrame([[Name,acc,BER,ROC,acc_std,BER_std,ROC_std]],columns=col)
        df=df.append(df1)

    return df


def replace_outlier(x,min_whisker,max_whisker):
    if x<min_whisker or x>max_whisker:
        x=np.NaN
    return x

def fill_outlier(df_ada,method):
    numerical_cols=df_ada.columns[df_ada.nunique()>2]
    for col in numerical_cols:
        q1=df_ada[col].quantile(0.25)
        q3=df_ada[col].quantile(0.75)
        IQR=q3-q1
        min_whisker=q1-1.5*IQR
        max_whisker=q3+1.5*IQR
        df_ada[col]=df_ada[col].apply(lambda x: replace_outlier(x,min_whisker,max_whisker))

    if method=="mean":
        df_ada=df_ada.fillna(df_ada.mean())
        
    if method=="knn":
        imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
        df_ada[numerical_cols]=imputer.fit_transform(df_ada[numerical_cols])
    return df_ada


def decisionTreeModel(x_train,y_train, x_test, y_test):
    params = {'max_depth': range(2,15),
         'min_samples_split': range(2,5),
         'min_samples_leaf': [1,2]}

    clf = tree.DecisionTreeClassifier()
    gcv = GridSearchCV(estimator=clf,param_grid=params)
    gcv.fit(x_train,y_train)
    GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=params )
    model = gcv.best_estimator_
    model.fit(x_train,y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    print(f'Train score {accuracy_score(y_train_pred,y_train)}')
    print(f'Test score {accuracy_score(y_test_pred,y_test)}')


def main():
    ada_train_data, ada_train_label, ada_valid_data, ada_valid_label, ada_test_data = loadData()
    ada_train_data=ada_train_data.drop(columns=['X21'])
    ada_valid_data=ada_valid_data.drop(columns=['X21'])
    ada_test_data=ada_test_data.drop(columns=['X21'])
    print(ada_train_data.isna().any().sum(),ada_valid_data.isna().any().sum(),ada_test_data.isna().any().sum())
    # ada_train_data= transformData(ada_train_data)
    # ada_train_label= transformData(ada_train_label)
    # ada_valid_data= transformData(ada_valid_data)
    # ada_valid_label= transformData(ada_valid_label)
    # ada_test_data= transformData(ada_test_data)

    print("score without any treatment")
    decisionTreeModel(ada_train_data, ada_train_label, ada_valid_data, ada_valid_label)

    # deleting outlayers
    print("score with deleting outlayers")
    ada_train_data_reduced  =ada_train_data.drop(detectOutLayers(ada_train_data), axis=0)
    ada_train_label_reduced  =ada_train_label.drop(detectOutLayers(ada_train_data), axis=0)
    ada_valid_data_reduced =ada_valid_data.drop(detectOutLayers(ada_valid_data), axis=0)
    ada_valid_label_reduced=ada_valid_label.drop(detectOutLayers(ada_valid_data), axis=0)
    decisionTreeModel(ada_train_data_reduced, ada_train_label_reduced, ada_valid_data_reduced , ada_valid_label_reduced)
    
    # filling outlayers with knn
    print("filling outlayers with knn methode")
    ada_train_data_knnFill =fill_outlier(ada_train_data,method="knn")
    ada_valid_data_knnFill =fill_outlier(ada_valid_data,method="knn")
    decisionTreeModel(ada_train_data_knnFill, ada_train_label, ada_valid_data_knnFill, ada_valid_label)

    # filling outlayers with mean
    # print("filling outlayers with mean methode")
    # ada_train_data_meanFill =fill_outlier(ada_train_data,method="mean")
    # ada_valid_data_meanFill =fill_outlier(ada_valid_data,method="mean")
    # decisionTreeModel(ada_train_data_meanFill, ada_train_label, ada_valid_data_meanFill , ada_valid_label)

    #  with standardisation
    print("accuraci with standardisation")
    ada_train_data_standard=normalize_num_cols(ada_train_data)
    ada_valid_data_standard=normalize_num_cols(ada_valid_data)
    decisionTreeModel(ada_train_data_standard, ada_train_label, ada_valid_data_standard, ada_valid_label)


    

if __name__ == '__main__':
    main()