import numpy as np
import os
import pandas as pd
from sklearn import metrics
# for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# modelling
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, PrecisionRecallDisplay, RocCurveDisplay, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit, cross_val_score
from surprise import Dataset,Reader
from surprise.model_selection import train_test_split
from surprise import SVD, KNNWithMeans
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

def loadData():
    books = pd.read_csv("BX-Books.csv", sep=";", error_bad_lines=False, encoding="latin-1")
    books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
    
    users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    users.columns = ['userID', 'Location', 'Age']

    ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    ratings.columns = ['userID', 'ISBN', 'bookRating']
    books = books[(books.yearOfPublication != 'DK Publishing Inc') & (books.yearOfPublication != 'Gallimard')]
    books['yearOfPublication'] = books['yearOfPublication'].astype('int32')
    books = books.dropna(subset=['publisher'])
    

# ## Exploring Users datase
    users.loc[(users.Age > 90) | (users.Age < 5), 'Age'] = np.nan
    users['Age'] = users['Age'].fillna(users['Age'].mean())
# ### Change the datatype of `Age` to `int`
    users['Age'] = users['Age'].astype(int)

# ## Exploring the Ratings Dataset
# ### Ratings dataset should have books only which exist in our books dataset. Drop the remaining rows
    ratings_new = ratings[ratings.ISBN.isin(books.ISBN)]
# ### Ratings dataset should have ratings from users which exist in users dataset. Drop the remaining rows
    ratings_new = ratings_new[ratings.userID.isin(users.userID)]
#Hence segragating implicit and explict ratings datasets
    ratings_explicit = ratings_new[ratings_new.bookRating != 0]
    ratings_implicit = ratings_new[ratings_new.bookRating == 0]
    
    counts1 = pd.value_counts(ratings_explicit['userID'])
    # ratings_explicit_new = ratings_explicit[ratings_explicit['userID'].isin(counts1[counts1 >= quantile(0.90)].index)]
    ratings_explicit_new = ratings_explicit[ratings_explicit['userID'].isin(counts1[counts1 >= 500].index)]
    ratings_explicit = ratings_explicit_new.copy()

    return (books, users, ratings, ratings_new, ratings_explicit, ratings_implicit)

def transformData(ratings_explicit):
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings_explicit[['userID', 'ISBN', 'bookRating']], reader)
    print(data.df.head(2))
    return data


def simpleRecoSystem(books, users, ratings, userID = "none"):
    ratings_explicit = ratings[ratings.bookRating != 0]
    ratings_explicit.to_csv("test.csv", sep=",", index=False)
    if not os.path.isfile("book_reco.csv"):
        
        books_ratings = {"ISBN":  [], "title":[], "nb_vote":[], "meanRating":[], "author":[], "publisher":[]}
        for bookID in ratings_explicit.ISBN.unique():
            bookInfo = books[books.ISBN == bookID]
            books_ratings ["ISBN"].append(bookID)
            books_ratings ["title"].append(bookInfo.bookTitle.values[0])
            books_ratings ["nb_vote"].append(len(ratings_explicit[ratings_explicit.ISBN == bookID]))
            books_ratings ["meanRating"].append(ratings_explicit[ratings_explicit.ISBN == bookID].bookRating.mean())
            books_ratings ["author"].append(bookInfo.bookAuthor.values[0])
            books_ratings ["publisher"].append(bookInfo.publisher.values[0])
        
        recoDf = pd.DataFrame.from_dict(books_ratings )
        recoDf.to_csv("book_reco.csv", sep=",", index=False)
    else:
        recoDf = pd.read_csv("book_reco.csv")
    m = recoDf.nb_vote.quantile(0.90)
    avalableReco = recoDf[recoDf.nb_vote >= m]
    avalableReco = avalableReco.assign(score = (avalableReco.nb_vote/(avalableReco.nb_vote+m) * avalableReco.nb_vote.mean()) + (m/(m + avalableReco.nb_vote) * avalableReco.meanRating))
    # if userID != "none" and userID in ratings_explicit.userID.tolist():
    if userID != "none":
        userBooks = ratings_explicit[ratings_explicit.userID == userID].ISBN.unique().tolist()
        print(userBooks)
        print(len(avalableReco))
        avalableReco = avalableReco[~avalableReco.ISBN.isin(userBooks)]
        print(len(avalableReco))
    finalReco = avalableReco.sort_values('score', ascending=False)
    print(finalReco [["title", "meanRating", "score"]].head(15))
    
        
        
        
    
    
def svmModel(data):
    trainset, testset = train_test_split(data, test_size=.20)
    svd_model = SVD(n_factors=5)
    svd_model.fit(trainset)
    test_pred = svd_model.test(testset)
    print(test_pred[0:5])
    # compute RMSE
    print(accuracy.rmse(test_pred))
    
    
def knnMeansModel(data):
    trainset, testset = train_test_split(data, test_size=.20)
    # algo_i = KNNWithMeans(k=10, sim_options={ 'user_based': False})
    # algo_i.fit(trainset)
    param_grid = {
    'k': range(10,30),
    'sim_options': {
        'name': ['msd', 'cosine'],
        'min_support': [1, 5],
        'user_based': [False],
    },}
    gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse", "mae"], cv=30)
    gs.fit(trainset)
    algo = gs.best_estimator["rmse"]
    algo.fit(trainset)
    test_pred=algo.test(testset)
    print(accuracy.rmse(test_pred))
    
    
    
    

def main():
    books, users, ratings, ratings_new, ratings_explicit, ratings_implicit = loadData()
    data = transformData(ratings_explicit)
    # simpleRecoSystem(books, users, ratings_new)
    simpleRecoSystem(books, users, ratings_new, userID = 44842)
    # svmModel(data)
    # knnMeansModel(data)
if __name__ == '__main__':
    main()