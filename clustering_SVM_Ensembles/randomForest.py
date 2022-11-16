import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
# for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# modelling
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, PrecisionRecallDisplay, RocCurveDisplay, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def count_na(df, col):
    print(f"Null values in {col}: ", df[col].isna().sum())

def eda_base(df):
    print(df.info()) # We clearly see that apart from the feature `Outcome`, every other feature is numerical and continuous in nature.
    print(df.describe()) # The minimum value of the features `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` is 0. This is logically incorrect as these values cannot be 0
    #   check for null value
    for feat in df.columns: count_na(df, feat)
    # no NA value
    eda_exploring_visualisation(df)

def eda_exploring_visualisation(df):
    sns.set_style("darkgrid")
    sns.set_palette("viridis")
    # ### Analysis of Pregnancies
# 
# As observed, `Pregnancies` is a **Quantitative** feature. There are many plots to analyse these type of data. Histograms, Box plots and Violin plots, are useful to know how the data is distributed.

    fig1, ax1 = plt.subplots(1, 2, figsize=(20, 7))
    fig2, ax2 = plt.subplots(figsize=(20, 7))
    sns.histplot(data=df, x="Pregnancies", kde=True, ax=ax1[0])
    sns.boxplot(data=df, x="Pregnancies", ax=ax1[1])
    sns.violinplot(data=df, x="Pregnancies", ax=ax2)
    plt.show()
    print("Median of Pregnancies: ", df["Pregnancies"].median())
    print("Maximum of Pregnancies: ", df["Pregnancies"].max())
    df["Pregnancies"].value_counts()

# From the above analysis we observe that:
# - Most patients had 0, 1 or 2 pregnancies.
# - Median value of `Pregnancies` is **3**.
# - Also, patients had upto **17** pregnancies!
# There are 3 outliers on the boxplot. But, let's not remove them for now.

# ### Analysis of Outcome (Target Variable)
# A Count plot and a Pie chart will be two useful plots to analyse the `Outcome` column as it is a categorical feature. Usefulness in the sense, both the plots will allow us to observe the distribution of each category in the feature.

    fig, ax = plt.subplots(1, 2, figsize=(20, 7))

    sns.countplot(data=df, x="Outcome", ax=ax[0])
    df["Outcome"].value_counts().plot.pie(explode=[0.1, 0], autopct="%1.1f%%", labels=["No", "Yes"], shadow=True, ax=ax[1])
    plt.show()
# We observe from the above plot that:
# - **65.1% patients in the dataset do NOT have diabetes.**
# - **34.9% patients in the dataset has diabetes.**

# ### Analysis of Glucose
# `Glucose` is a **Quantitative** feature. Histograms, Box plots and Violin plots, are useful to know how the data is distributed.

    fig3, ax3 = plt.subplots(1, 2, figsize=(20, 7))
    fig4, ax4 = plt.subplots(figsize=(20, 7))

    sns.histplot(data=df, x="Glucose", kde=True, ax=ax3[0])
    sns.boxplot(data=df, x="Glucose", ax=ax3[1])
    sns.violinplot(data=df, x="Glucose", ax=ax4)
    plt.show()
    print("Median of Glucose: ", df["Glucose"].median())
    print("Maximum of Glucose: ", df["Glucose"].max())
    print("Mean of Glucose: ", df["Glucose"].mean())
    print("Rows with Glucose value of 0: ", df[df["Glucose"] == 0].shape[0])

# We observe that:
# - Median (117.0) and mean (120.8) of `Glucose` lie very close to each other i.e. the distribution is more or less **symmetric and uniform**.
# - As seen from the box plot, an outlier lies on 0-value, which I talked about earlier.
# - There are **5 rows** with `Glucose` value as 0. This is not logical, so we need to keep this in mind.

# ### Analysis of Blood Pressure
# `BloodPressure` is a **Quantitative** feature. Histograms, Box plots and Violin plots, are useful to know how the data is distributed.

    fig5, ax5 = plt.subplots(1, 2, figsize=(20, 7))
    fig6, ax6 = plt.subplots(figsize=(20, 7))
    sns.histplot(data=df, x="BloodPressure", kde=True, ax=ax5[0])
    sns.boxplot(data=df, x="BloodPressure", ax=ax5[1])
    sns.violinplot(data=df, x="BloodPressure", ax=ax6)
    plt.show()
    print("Median of Blood Pressure: ", df["BloodPressure"].median())
    print("Maximum of Blood Pressure: ", df["BloodPressure"].max())
    print("Mean of Pressure: ", df["BloodPressure"].mean())
    print("Rows with BloodPressure value of 0: ", df[df["BloodPressure"] == 0].shape[0])
# We observe that:
# - Median (72.0) and mean (69.1) of `BloodPressure` lie very close to each other i.e. the distribution is more or less **symmetric and uniform**.
# - As seen from the box plot and violin plot, some outliers lie on 0-value, which I talked about earlier.
# - There are **35 rows** with `BloodPressure` value as 0. This is not logical.
# ### Analysis of Insulin
# 
# Plotting Histogram, Box plot and Violin plot for `Insulin`.

    fig7, ax7 = plt.subplots(1, 2, figsize=(20, 7))
    fig8, ax8 = plt.subplots(figsize=(20, 7))
    sns.histplot(data=df, x="Insulin", kde=True, ax=ax7[0])
    sns.boxplot(data=df, x="Insulin", ax=ax7[1])
    sns.violinplot(data=df, x="Insulin", ax=ax8)
    plt.show()

    print("Rows with Insulin value of 0: ", df[df["Insulin"] == 0].shape[0])

# The plots for `Insulin` are highly skewed. Also, the 0-value logical error is the most for this feature. **374 out of 768** instances have value of `Insulin` as 0.

# ### Analysis of BMI
#   Plotting Histogram, Box plot and Violin plot for `BMI`.
    fig9, ax9 = plt.subplots(1, 2, figsize=(20, 7))
    fig10, ax10 = plt.subplots(figsize=(20, 7))

    sns.histplot(data=df, x="BMI", kde=True, ax=ax9[0])
    sns.boxplot(data=df, x="BMI", ax=ax9[1])
    sns.violinplot(data=df, x="BMI", ax=ax10)
    plt.show()
    print("Median of BMI: ", df["BMI"].median())
    print("Maximum of BMI: ", df["BMI"].max())
    print("Mean of BMI: ", df["BMI"].mean())
    print("Rows with BMI value of 0: ", df[df["BMI"] == 0].shape[0])
# We observe that:
# - Median (32.0) and Mean (31.9) of `BMI` are very close to each other. Thus, the distribution is more or less **symmetric and uniform**
# - Maximum BMI is 67.1
# - There are **11 rows** with `BMI` value as 0
# ### Analysis of Diabetes Pedigree Function
# `DiabetesPedigreeFunction` is a **continuous and quantitative** variable.

    fig11, ax11 = plt.subplots(1, 2, figsize=(20, 7))
    
    fig12, ax12 = plt.subplots(figsize=(20, 7))
    sns.histplot(data=df, x="DiabetesPedigreeFunction", kde=True, ax=ax11[0])
    sns.boxplot(data=df, x="DiabetesPedigreeFunction", ax=ax11[1])
    sns.violinplot(data=df, x="DiabetesPedigreeFunction", ax=ax12)
    plt.show()
    print("Median of DiabetesPedigreeFunction: ", df["DiabetesPedigreeFunction"].median())
    print("Maximum of DiabetesPedigreeFunction: ", df["DiabetesPedigreeFunction"].max())
    print("Mean of DiabetesPedigreeFunction: ", df["DiabetesPedigreeFunction"].mean())

# We observe that:
# - The histogram is higly skewed on the left side.
# - There are many outliers in the Box plot.
# - Violin plot distribution is dense in the interval `0.0 - 1.0`

# ### Analysis of Age
# Plotting Histogram, Box plot and Violin plots for `Age`.
    fig13, ax13 = plt.subplots(1, 2, figsize=(20, 7))
    fig14, ax14 = plt.subplots(figsize=(20, 7))
    sns.histplot(data=df, x="Age", kde=True, ax=ax13[0])
    sns.boxplot(data=df, x="Age", ax=ax13[1])
    sns.violinplot(data=df, x="Age", ax=ax14)
    plt.show()
    print("Median of Age: ", df["Age"].median())
    print("Maximum of Age: ", df["Age"].max())
    print("Mean of Age: ", df["Age"].mean())
# We again observe that:
    # - The distribution of Age is skewed on the left side.
    # - There are some outliers in the Box plot for Age.
    # ### Analysis of Glucose and Outcome
# Since `Glucose` is a continuous feature, we plot a histogram with its hue based on `Outcome`.
    fig15, ax15 = plt.subplots(figsize=(20, 8))
    sns.histplot(data=df, x="Glucose", hue="Outcome", shrink=0.8, multiple="fill", kde=True, ax=ax15)
    plt.show()
# From the above plot, we see a **positive linear correlation**.
# - As the value of `Glucose` increases, the count of patients having diabetes increases i.e. value of `Outcome` as 1, increases.
# - Also, after the `Glucose` value of **125**, there is a steady increase in the number of patients having `Outcome` of 1.
# - Note, when `Glucose` value is 0, it means the measurement is missing. We need to fill that values with the *mean* or *median* and then it will make sense.
# 
# So, there is a significant amount of *positive* linear correlation.

# ### Analysis of BloodPressure and Outcome
# `BloodPressure` is continuous and `Outcome` is binary feature. So, plotting a histogram for `BloodPressure` with its hue based on `Outcome`.
    fig16, ax16 = plt.subplots(figsize=(20, 8))
    sns.histplot(data=df, x="BloodPressure", hue="Outcome", shrink=0.8, multiple="dodge", kde=True, ax=ax16)
    plt.show()
# We observe that, `Outcome` and `BloodPressure` do **NOT** have a positive or negative linear correlation. The value of `Outcome` do not increase linearly as value of `BloodPressure` increases.
# However, for `BloodPressure` values greater than 82, count of patients with `Outcome` as 1, is more.

# ### Analysis of BMI and Outcome
    fig17, ax17 = plt.subplots(figsize=(20, 8))
    sns.histplot(data=df, x="BMI", hue="Outcome", shrink=0.8, multiple="fill", kde=True, ax=ax17)
    plt.show()

# From the above plot, a **positive linear correlation** is evident for `BMI`.
# ### Analysis of Age and Outcome
# `Age` is continuous so plotting a histogram with hue based on `Outcome`.
    fig18, ax18 = plt.subplots(figsize=(20, 8))
    sns.histplot(data=df, x="Age", hue="Outcome", shrink=0.8, multiple="dodge", kde=True, ax=ax18)
    plt.show()
# For `Age` greater than 35 years, the chances of patients having diabetes increases as evident from the plot i.e. The number of patients having diabetes is more than the number of people **NOT** having diabetes. But, it does not hold true for ages like **60+**, somehow.
# There is *some* positive linear correlation though.

# ### Analysis of Pregnancies and Outcome
    fig19, ax19 = plt.subplots(figsize=(20, 8))
    sns.histplot(data=df, x="Pregnancies", hue="Outcome", shrink=0.8, multiple="fill", kde=True, ax=ax19)
    plt.show()
# There is *some* positive linear correlation of `Pregnancies` with `Outcome`.

# Let us plot a **heatmap** of the correlation matrix of different features.
# The 2D correlation matrix
    corr_matrix = df.corr()

#   Plotting the heatmap of corr
    fig20, ax20 = plt.subplots(figsize=(20, 7))
    dataplot = sns.heatmap(data=corr_matrix, annot=True, ax=ax20)
    plt.show()

    corr_matrix["Outcome"].sort_values(ascending=False)

# We observe that:
# - `Glucose` has the maximum positive linear correlation with `Outcome`, which is logical.
# - `BloodPressure` has the lowest positive linear correlation with `Outcome`.
# - No feature has a negative linear correlation with `Outcome`.

def replace_with_median(ndf, feat, value):
    ndf[feat] = ndf[feat].replace(0, value)
def get_glucose_proportions(ndf):
    print(ndf["Glucose_cat"].value_counts() / len(ndf))
    
    
def transformData(df, applyDimentionReduction = False):
    newdf = df
    newdf["Glucose_cat"] = pd.cut(newdf["Glucose"],
                           bins=[-1, 40, 80, 120, 160, np.inf],
                            labels=[1, 2, 3, 4, 5])
    print(newdf["Glucose_cat"].value_counts())
    fig21, ax21 = plt.subplots(figsize=(20, 7))
    newdf["Glucose_cat"].hist(ax=ax21)
    plt.show()
    # first, spliting the data
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    
    #  feature ingeniering
    for train_index, test_index in split.split(newdf, newdf["Glucose_cat"]):
        strat_train_set = newdf.loc[train_index]
        strat_test_set = newdf.loc[test_index]


# We can now compare the proportions of various `Glucose` categories between the Testing set and the entire dataset.
# The output below shows the various proportions.

    print("Entire Dataset: ")
    get_glucose_proportions(newdf)
    print("\n")
    print("-"*30)
    print("\nTesting set: ")
    get_glucose_proportions(strat_test_set)


# We now drop the `Glucose_cat` column to bring the data back to its original form.
    for set_ in (strat_train_set, strat_test_set):
        set_.drop(columns=["Glucose_cat"], inplace=True)

    # dealing with missing value (0 value)
    meds = []
    feats = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for feat in feats:
        meds.append(strat_train_set[feat].median())
    print("Medians are: ", meds)
# helper function
    
    for i, feat in enumerate(feats):
        replace_with_median(strat_train_set, feat, meds[i])
        replace_with_median(strat_test_set, feat, meds[i])


# With this, we replaced all the missing values with the median (of that column) in the Training set and used those *learned* medians to replace the missing values in the Testing set.
    X_train = strat_train_set.drop(columns="Outcome")
    y_train = strat_train_set["Outcome"]
    X_test = strat_test_set.drop(columns="Outcome")
    y_test = strat_test_set["Outcome"]
    
    # scalling the data
    # 
# Since the ranges of different features vary too much, it is necessary to *scale* them so that the Machine Learning models can perform even better. 
# We will use **Scikit-Learn's Standard Scaler** to scale the features.
    labels = X_test.columns
    stdscaler = StandardScaler()
    stdscaler.fit(X_train)
    X_train_ = stdscaler.transform(X_train)
    X_test_ = stdscaler.transform(X_test)
    print("Scaled training set: ", X_train_)
    print("Scaled testing set: ", X_test_)
    if applyDimentionReduction :
        X_train_ = get_umap(X_train_)
        X_test_ = get_umap(X_test_)
    return  (X_train_, X_test_, y_train, y_test, labels)
    
    
def get_umap(df):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(df)
    print(embedding.shape)
    print(embedding)
    return embedding
    # ploting the new shape of data
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[x] for x in df.species.map({"Adelie":0, "Chinstrap":1, "Gentoo":2})])
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of the diabet  dataset', fontsize=24)
    # plt.show()


def get_random_forests(X_train, X_test, y_train, y_test, labels):
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# feature importance
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)
    feature_imp = pd.Series(clf.feature_importances_,index=labels).sort_values(ascending=False)
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()



def main():
    df =  pd.read_csv("diabetes.csv")
    print(df)
    # eda_base(df)
    X_train_, X_test_, y_train, y_test, labels = transformData(df)
    X_train_reduce, X_test_reduce, y_train_reduce, y_test_reduce, labels = transformData(df, applyDimentionReduction =True)
    get_random_forests(X_train_, X_test_, y_train, y_test , labels)
    get_random_forests(X_train_reduce, X_test_reduce, y_train_reduce, y_test_reduce, ["val1", "val2"])




if __name__ == '__main__':
    main()