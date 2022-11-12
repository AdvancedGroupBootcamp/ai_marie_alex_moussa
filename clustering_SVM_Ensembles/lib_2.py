# import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

def eda_base(df, correlation = 0, n_round = 2):
	display(df.head())
	display(df.info())
	
	print('isNa?', df.isna().any().sum())
	display(df.describe().round(n_round))


def plot_corr(title = '', x_rotation = 45):
	# correlation - by Heatmap the relationship between the features.
	plt.figure(figsize=(10,6))
	sns.heatmap(df.corr(), cmap=plt.cm.Reds, annot=True)
	plt.yticks(rotation = 0) 
	plt.xticks(rotation = x_rotation) 
	plt.title(title, fontsize=13)
	plt.show()


def plot_hist(df):
	# distributions of the variables/features.
	df.hist(figsize=(12,8),bins=20)
	plt.show()

