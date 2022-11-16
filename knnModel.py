import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col
# import for knn model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.feature_selection import chi2,SelectKBest,RFE,f_classif
import warnings

warnings.filterwarnings('ignore')


def encodeData(train_df, test_df):
    # Test data
    test_df['Pclass']=test_df['Pclass'].astype('category')
    test_df['Cabin_new']=test_df['Cabin_new'].apply(lambda x:str(x).strip()[0])
    test_df_numeric=pd.get_dummies(test_df)
    # train data
    train_df['Pclass']=train_df['Pclass'].astype('category')
    train_df['Cabin_new']=train_df['Cabin_new'].apply(lambda x:str(x).strip()[0])
    train_df[train_df["Fare"]==35]
    train_df['Cabin_new']=train_df['Cabin_new'].replace('T','C') 
    train_df_numeric=pd.get_dummies(train_df)
    return train_df_numeric, test_df_numeric

def buildKNNModel(df, targetColumn, metricsList, sampleSize):
    # definding our variable to train
    y= df[targetColumn]
    x= df[metricsList]
    # preparation of model
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=sampleSize,  random_state=0)

    # optimisation test
    KNN = KNeighborsClassifier()
    k_range = list(range(1, 31))
    param_grid = dict(n_neighbors=k_range)
    # defining parameter range
    grid = GridSearchCV(KNN, param_grid, cv=10, scoring='accuracy', return_train_score=False,verbose=1)
  
    # fitting the model for grid search
    grid_search=grid.fit(x_train,y_train)


    print(grid_search.best_params_)

    # checking the model with the optimal parameter
    knn = KNeighborsClassifier(n_neighbors= int(grid_search.best_params_['n_neighbors']))
    # now, we train our model
    knn.fit(x_train,y_train)

    # we check the accuracy or our model
    print(knn.score(x_valid,y_valid))
    print("Accuracy for our training dataset with tuning is : {:.2%}".format(grid_search.best_score_) )

    #  prediction
    y_predict=grid_search.predict(x_valid)
    print(classification_report(y_valid,y_predict))
    shufflesplit=ShuffleSplit(n_splits=10, test_size=sampleSize,random_state=0)
    results=cross_val_score(grid_search, x_valid, y_valid, cv=shufflesplit)
    print(f"The average score on the model's predictive ability:{np.mean(results):.2%} with std:{np.std(results):.2%}")

    # getting the most important variable
    # selector=SelectKBest(score_func=f_classif)
    # selector_fitted=selector.fit(x_train, y_train)
    # print("Significant variables in modeling passenger survival: ",x_train.columns[selector_fitted.get_support()])

    # confusion matrix
    y_valid_pred=grid_search.predict(x_valid)
    confu_mat= metrics.confusion_matrix(y_valid,y_valid_pred)
    cm_obj=metrics.ConfusionMatrixDisplay(confu_mat,
                                      #display_labels=randomforest.
                                     )
    plt.figure(figsize=(10,14))
    plt.rcParams.update({'font.size':15})
    cm_obj.plot()
    cm_obj.ax_.set(title='Confusion matrix')

    plt.show()

def main():
    train_df =pd.read_csv("train_processed.csv")
    test_df =pd.read_csv("test_processed.csv")
    train_df, test_df = encodeData(train_df, test_df)
    attributList = [
        ['Fare', 'Sex_female', 'Sex_male'],
        ['Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male'],
        ['Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_S'],
        ['Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_S', 'Cabin_new_C', 'Cabin_new_F'],
        # ['Age'], ['SibSp'], ['Parch'], ['Fare'], 
        # ['Pclass_1', 'Pclass_2', 'Pclass_3'], 
        # ['Sex_female', 'Sex_male'], 
        # ['Embarked_C', 'Embarked_Q', 'Embarked_S'], 
        # ['Cabin_new_A', 'Cabin_new_B', 'Cabin_new_C', 'Cabin_new_D', 'Cabin_new_E', 'Cabin_new_F', 'Cabin_new_G']
        ]
    

    for attributSet in attributList:
        print("running model for attribut set : "+  str(attributSet ))
        # buildKNNModel(train_df, "Survived", ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Cabin_new_A', 'Cabin_new_B', 'Cabin_new_C', 'Cabin_new_D', 'Cabin_new_E', 'Cabin_new_F', 'Cabin_new_G'], 0.1)
        buildKNNModel(train_df, "Survived", attributSet , 0.1)




if __name__ == '__main__':
    main()