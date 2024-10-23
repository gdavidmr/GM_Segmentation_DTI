# import modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# set working directory
DIR = 'D:/machine_learning/gm_segmentation_dti'
os.chdir(DIR)

# specify data source
data = os.path.join(DIR,'data_source.mat')


#===================== get the data =======================
# load source matfile into a dictionary
from data_prep_load import data_prep_load
dic = data_prep_load(data)

# convert the dictionary into a dataframe (df), where
# - rows represent observations (individual voxels)
# - columns represent the 4 feature variables and the single label
from data_prep_convert import data_prep_convert
df = data_prep_convert(dic)


# Important question: which performance metric to use?
# Since the classes are more or less balanced, accuracy
# is a good metric.
df["tissue"].value_counts()



# ================= CHECK DATA STRUCTURE ================

df.head(5)
df.info()
df.dtypes

# get the size of the df
print('number of dimensions of df: ' + str(df.ndim))
print('size of df: ' + str(df.size))
print('shape of df: ' + str(df.shape))

# summary measures of the numeric attributes
df.describe()
print(df.mean())
print(df.median())

# split df into training and test dataset
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)


# ================= EXPLORATORY DATA ANALYSIS ================

# EDA is carried out on the training set only!!!

# histograms
df_train.hist(bins=50)

# correlations
corr = df_train.corr()
# AD and RD highly correlate with MD (obviously) (r~0.9), but they also strongly correlate
# with each other (r~0.8). MD does not correlate with FA.
# AD and RD correlate with FA (abs(r)~0.5). 

# scatter plots
from pandas.plotting import scatter_matrix
scatter_matrix(df_train)

# plotting
idx_wm = np.array(np.where(df_train["tissue"]==0))
idx_gm = np.array(np.where(df_train["tissue"]==1))

plt.figure(figsize=(10,10))
plt.plot(df_train.loc[idx_wm[0,:5000],"fa"], df_train.loc[idx_wm[0,:5000],"md"], "b.", linewidth=0.01, alpha=0.1, label="WM")
plt.plot(df_train.loc[idx_gm[0,:5000],"fa"], df_train.loc[idx_gm[0,:5000],"md"], "r.", linewidth=0.01, alpha=0.1, label="GM")
plt.xlabel("FA")
plt.ylabel("MD")
plt.legend()
plt.savefig( os.path.join(DIR,'figure_fa_md.png'))

plt.figure(figsize=(10,10))
plt.plot(df_train.loc[idx_wm[0,:5000],"fa"], df_train.loc[idx_wm[0,:5000],"ad"], "b.", linewidth=0.01, alpha=0.1, label="WM")
plt.plot(df_train.loc[idx_gm[0,:5000],"fa"], df_train.loc[idx_gm[0,:5000],"ad"], "r.", linewidth=0.01, alpha=0.1, label="GM")
plt.xlabel("FA")
plt.ylabel("AD")
plt.legend()
plt.savefig( os.path.join(DIR,'figure_fa_ad.png'))

plt.figure(figsize=(10,10))
plt.plot(df_train.loc[idx_wm[0,:5000],"fa"], df_train.loc[idx_wm[0,:5000],"rd"], "b.", linewidth=0.01, alpha=0.1, label="WM")
plt.plot(df_train.loc[idx_gm[0,:5000],"fa"], df_train.loc[idx_gm[0,:5000],"rd"], "r.", linewidth=0.01, alpha=0.1, label="GM")
plt.xlabel("FA")
plt.ylabel("RD")
plt.legend()
plt.savefig( os.path.join(DIR,'figure_fa_rd.png'))

plt.figure(figsize=(10,10))
plt.plot(df_train.loc[idx_wm[0,:5000],"ad"], df_train.loc[idx_wm[0,:5000],"rd"], "b.", linewidth=0.01, alpha=0.1, label="WM")
plt.plot(df_train.loc[idx_gm[0,:5000],"ad"], df_train.loc[idx_gm[0,:5000],"rd"], "r.", linewidth=0.01, alpha=0.1, label="GM")
plt.xlabel("AD")
plt.ylabel("RD")
plt.legend()
plt.savefig( os.path.join(DIR,'figure_ad_rd.png'))



# ================= DATA PROCESSING ================

# Notes:
# - we only process the training data at this point
# - need to split the training data into a feature matrix and label vector, as they need separate processing
# - processing steps should be combined into pipelines
# - numerical and categorical attributes require separate pipeline

# In this dataset, we only have 4 numerical attributes, so one pipeline is sufficient.

# split the dataframe into a feature matrix and label vector
X_train = df_train.drop(columns=['tissue'])
y_train = df_train['tissue'].copy()

# dealing with missing values: no missing values in the dataset
df_train.info()

# feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

pickle_out = open(os.path.join(DIR,"scaler.pickle"),"wb")
pickle.dump(scaler,pickle_out)
pickle_out.close()


# just for fun: PCA
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_train)
print('PCA loadings: ' + '\n' + str(pca.components_)) # loadings
#pca. # loadings
pca.explained_variance_ # total explained variance
pca.explained_variance_ratio_ # relative explained variance
# Actually, two variables explain most of the variance: extract reduced data
X_train_pca = pca.transform(X_train)




# ================= SELECTING CLASSIFIER ================

# start with Logistic regression: it's a linear classifier (linear decision boundary)
# apply a moderately large regularization (C=0.1)
from sklearn.linear_model import LogisticRegression
clf_logreg = LogisticRegression(C=0.1)
clf_logreg.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
scores_clf_logreg = cross_val_score(clf_logreg, X_train, y_train, cv=10, scoring="accuracy")
score_clf_logreg = scores_clf_logreg.mean()
print("Cross-validated accuracy: " + str(scores_clf_logreg.mean()))
# 88%, not bad for a linear classifier!!! But maybe we can do better

pickle_out = open(os.path.join(DIR,"clf_logreg.pickle"),"wb")
pickle.dump(clf_logreg,pickle_out)
pickle_out.close()



# next, try a linear SVM classifier
from sklearn.svm import LinearSVC
clf_svm_lin = LinearSVC(C=0.1)
clf_svm_lin.fit(X_train, y_train)

scores_clf_svm_lin = cross_val_score(clf_svm_lin, X_train, y_train, cv=10, scoring="accuracy")
score_clf_svm_lin = scores_clf_svm_lin.mean()
print("Cross-validated accuracy: " + str(scores_clf_svm_lin.mean()))
# this is still pretty much the same 88%, but it's not surprising given it's a linear classifier as well






# try a non-linear classifier: apply the kernel trick
# in sklearn, only the SVC class supports the kernel trick for SVM classification
from sklearn.svm import SVC
clf_svm = SVC(kernel="rbf", gamma=0.1, C=0.1)
clf_svm.fit(X_train, y_train)

# accuracy on the training set (out of curiosity, doesn't make much sense)
from sklearn.metrics import accuracy_score
accuracy_score(y_train, clf_svm.predict(X_train))

# cross-validated accuracy
scores_clf_svm = cross_val_score(clf_svm, X_train, y_train, cv=10, scoring="accuracy")
score_clf_svm = scores_clf_svm.mean()
print("Cross-validated accuracy: " + str(scores_clf_svm.mean()))
# Even with this classifier, we got an around 88% accuracy, even a slightly lower value

# grid search:
from sklearn.model_selection import GridSearchCV
param_grid_clf_svm = [{"gamma":[0.1,1,10], "C":[0.1,1,10]}]
grid_search_clf_svm = GridSearchCV(clf_svm, param_grid_clf_svm, cv=10, scoring="accuracy", verbose=2)
grid_search_clf_svm.fit(X_train, y_train)
grid_search_clf_svm.best_estimator_
print(grid_search_clf_svm.best_score_)
pickle_out = open(os.path.join(DIR,"clf_svm.pickle"),"wb")
pickle.dump(grid_search_clf_svm,pickle_out)
pickle_out.close()
# Interestingly, the best non-linear SVM classifier returns the same accuracy as the linear classifiers.

# Now stay with the best-performing non-linear SVM, but train it on the PCA reduced dataset
scores_clf_svm = cross_val_score(grid_search_clf_svm.best_estimator_, X_train_pca, y_train, cv=10, scoring="accuracy")
score_clf_svm = scores_clf_svm.mean()
print("Cross-validated accuracy: " + str(scores_clf_svm.mean()))
# Still the same result







# try an even more complex model: a random forest classifier
from sklearn.ensemble import RandomForestClassifier
clf_forest = RandomForestClassifier(n_estimators=10)
clf_forest.fit(X_train, y_train)

scores_clf_forest = cross_val_score(clf_forest, X_train, y_train, cv=10, scoring="accuracy")
score_clf_forest = scores_clf_forest.mean()
print("Cross-validated accuracy: " + str(scores_clf_forest.mean()))






# ================= GET GENERALIZATION ERROR ================
X_test = df_test.drop(columns=['tissue'])
y_test = np.array(df_test["tissue"].copy())
X_test = scaler.transform(X_test)
y_test_predict = grid_search_clf_svm.best_estimator_.predict(X_test)
print("Generalization error: " + str(accuracy_score(y_test, y_test_predict)))

# Note that the accuracy is the same as in the training set -> the model is not overfitting due to regularization


