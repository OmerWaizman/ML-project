import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import csv
from sklearn import datasets
import plotly.express as px
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, plot_confusion_matrix, roc_auc_score, roc_curve, plot_roc_curve, classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn import tree
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay


df = pd.read_csv("C:\\Users\\Owner\\Downloads\\XY_train.csv")

#################################### cleaning the dataset ########################
df['company_size'] = df['company_size'].replace('10/49', '10-49')
df['experience'] = df['experience'].replace('>20', 21)
df['experience'] = df['experience'].replace('<1', 0)
df['last_new_job'] = df['last_new_job'].replace('>4',5)
df['last_new_job'] = df['last_new_job'].replace('never',0)

################### add values to enrolled_university column with null ############################

df['enrolled_university'].fillna('no_enrollment', inplace=True)

################### add values to edcuation level column with null ############################

df['education_level'].fillna('High School', inplace=True)

################# add values to major discipline column with null ##############################
for i in df['education_level']:
    if i == 'Graduate' or i == 'Masters':
        df['major_discipline'].fillna('STEM', inplace=True)
    elif i == 'High School' or i == 'Primary School':
        df['major_discipline'].fillna('No Major', inplace=True)

################### last new job null values #####################
df['last_new_job'].fillna(int(1), inplace=True)
df['last_new_job'] = df['last_new_job'].astype('int64')
#print(df['last_new_job'].value_counts())

################# gender fill missing values ##################
df['gender'].fillna('Male', inplace= True)

######################## remove experience with null values        ##################
df = df.dropna(subset = ['experience'])
# print(df['city_development_index'][0:10])
#################################### fill company_size####################
df['company_size'].fillna('50-99', inplace = True)

########################## fill company_type #######################

df['company_type'].fillna('Pvt Ltd',inplace= True)

#################### ordinal categories ################333
df['education_level'] = df['education_level'].map({'Primary School':int(0),'Graduate':int(2) ,'High School':int(1),
                                                                   'Phd':int(4),'Masters':int(3)})
df['enrolled_university'] = df['enrolled_university'].map({'no_enrollment':int(0),'Part time course':int(1) ,'Full time course':int(2),})

##################### normalzation #######################
df['experience'] = df['experience'].astype('float64')
df['training_hours'] = (df['training_hours'] - min(df['training_hours'])) / (max(df['training_hours']) - min(df['training_hours']))
df['experience'] = (df['experience'] - min(df['experience'])) / (max(df['experience']) - min(df['experience']))

df_improve = pd.DataFrame(df)
################### dummies ###########################
df_dummies = pd.DataFrame(df,columns=['relevent_experience','company_size','company_type'])
new_df = pd.get_dummies(df_dummies)
df = df.drop(columns=(['city','gender','enrollee_id','relevent_experience','major_discipline','company_size','company_type']))
df = pd.concat([df, new_df], axis=1)

#------------------------------------------------ Part B ----------------------------------------------------------

x = df.drop('target',axis=1)
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

########### Full desicion tree model #############
# model = DecisionTreeClassifier()
# model.fit(x_train, y_train)
# training_score = roc_auc_score(y_test, model.predict_proba(x_test)[:,1])
# print(f"DT_ruc_auc:{training_score:.2f}")

##########        K_FOLD       ###############3
kf = KFold(n_splits=10, shuffle=True, random_state=123)
def train_model_by_kfold(df, model):
    kf = KFold(n_splits=10)
    X = df.drop(columns=['target'])
    y = df['target']
    scores = pd.DataFrame()
    for train_index, test_index in kf.split(df):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train,y_train)
        training_score = roc_auc_score(y_train, model.predict_proba(X_train)[:,1])
        validation_score = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        scores = scores.append({'Training Score': training_score, 'Validation Score': validation_score}, ignore_index=True)

    print('\n\nTraining accuracy score: %.3f +/- %.3f\n Validation accuracy score %.3f +/- %.3f' % (scores['Training Score'].mean(), scores['Training Score'].std(), scores['Validation Score'].mean(),  scores['Validation Score'].std()))
    return ''

# print(train_model_by_kfold(df,model))

# param_grid = {'max_depth': np.arange(1, 10, 1),
#               'criterion': ['entropy','gini'],
#               'max_features': ['auto', 'sqrt', 'log2', None],
#               'min_samples_leaf':np.arange(1, 10, 1),
#               'min_samples_split': np.arange(2,16,1)
#              }
# grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
#                            param_grid=param_grid,
#                            refit=True,
#                            cv=10,scoring='roc_auc')
#
# grid_search.fit(x_train, y_train)
# print(grid_search.best_params_)


# ############ Tuning ##########
# model_ht = DecisionTreeClassifier(criterion='gini',max_depth=5,max_features=None ,random_state=27, min_samples_leaf=8,min_samples_split= 2 )
# model_ht.fit(x_train, y_train)
# training_score = roc_auc_score(y_test, model_ht.predict_proba(x_test)[:,1])
# print(f"DT_ht_ruc_auc:{training_score:.2f}")
# print(train_model_by_kfold(df,model_ht))


########## max_depth graph ####################


# max_depth_list = np.arange(1, 15, 1)
# res = pd.DataFrame()
# for max_depth in max_depth_list :
#     model_max_depth = DecisionTreeClassifier(criterion='gini', max_depth=max_depth ,
#                                    min_samples_leaf=8, min_samples_split=2, random_state=27)
#     model_max_depth.fit(x_train, y_train)
#     val_score=cross_val_score(model_max_depth, x_test, y_test,scoring='roc_auc', cv=10).mean()
#     res = res.append({'max_depth': max_depth,
#                       'train_ruc_auc':roc_auc_score(y_test, model_max_depth.predict(x_test)),
#                       'val_ruc_auc':val_score}, ignore_index=True)
# plt.figure(figsize=(13, 4))
# plt.plot(res['max_depth'], res['train_ruc_auc'], marker='o', markersize=4)
# plt.plot(res['max_depth'], res['val_ruc_auc'], marker='o', markersize=4)
# plt.legend(['Train ruc_auc', 'Validation ruc_auc'])
# plt.show()



## ####                 min sample split graph               #######
# min_samples_split_list = np.arange(2, 41, 1)
# res = pd.DataFrame()
# for sample_split in min_samples_split_list :
#     model_samples_split = DecisionTreeClassifier(criterion='gini', max_depth=5 ,
#                                    min_samples_leaf=8, min_samples_split=sample_split, random_state=27)
#     model_samples_split.fit(x_train, y_train)
#     val_score=cross_val_score(model_samples_split, x_test, y_test,scoring='roc_auc', cv=10).mean()
#     predict = model_samples_split.predict_proba(x_test)
#     res = res.append({'min_samples_split': sample_split,
#                       'train_ruc_auc':roc_auc_score(y_test, predict[:,1]),
#                       'val_ruc_auc':val_score}, ignore_index=True)
# plt.figure(figsize=(13, 4))
# plt.plot(res['min_samples_split'], res['train_ruc_auc'], marker='o', markersize=4)
# plt.plot(res['min_samples_split'], res['val_ruc_auc'], marker='o', markersize=4)
# plt.legend(['Train ruc_auc', 'Validation ruc_auc'])
# plt.show()


#############             min sample leaf          #######################3

# min_samples_leaf_list = np.arange(1, 41, 1)
# res = pd.DataFrame()
# for sample_leaf in min_samples_leaf_list :
#     model_sample_leaf = DecisionTreeClassifier(criterion='gini', max_depth=5 ,
#                                    min_samples_leaf=sample_leaf, min_samples_split=2, random_state=27)
#     model_sample_leaf.fit(x_train, y_train)
#     val_score=cross_val_score(model_sample_leaf, x_test, y_test,scoring='roc_auc', cv=10).mean()
#     predict = model_sample_leaf.predict_proba(x_test)
#     res = res.append({'min_samples_leaf': sample_leaf,
#                       'train_ruc_auc':roc_auc_score(y_test, predict[:,1]),
#                       'val_ruc_auc':val_score}, ignore_index=True)
# plt.figure(figsize=(13, 4))
# plt.plot(res['min_samples_leaf'], res['train_ruc_auc'], marker='o', markersize=4)
# plt.plot(res['min_samples_leaf'], res['val_ruc_auc'], marker='o', markersize=4)
# plt.legend(['Train ruc_auc', 'Validation ruc_auc'])
# plt.show()


##################### Tuning_with_graph#############
model_ht = DecisionTreeClassifier(criterion='gini', max_depth=7, max_features=None ,random_state=27, min_samples_leaf=35,min_samples_split= 40 )
model_ht.fit(x_train, y_train)
# training_score = roc_auc_score(y_test, model_ht.predict_proba(x_test)[:,1])
# print(f"DT_ht_ruc_auc:{training_score:.2f}")
#
# print(train_model_by_kfold(df,model_ht))

############### tree graph #################
# plt.figure(figsize=(8, 8))
# plot_tree(model_ht, filled=True,max_depth=2, class_names=True, fontsize=10,feature_names=list(x_train.columns))
# plt.savefig('full_tree_high_dpi', dpi=5)
# plt.show()

################# feature importences ###############

# from matplotlib import pyplot
# dt_model = DecisionTreeClassifier(criterion='gini', max_depth=5,max_features=None ,random_state=27, min_samples_leaf=8,min_samples_split= 2)
# dt_model.fit(x_train, y_train)
# importance = dt_model.feature_importances_
# # summarize feature importance
# for i,v in enumerate(importance):
# 	print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# pyplot.bar([x for x in range(len(importance))], importance)
# pyplot.show()


############################ ANN #####################

# ann_modle = MLPClassifier(random_state=1)
# ann_modle.fit(x_train, y_train)
# predict = ann_modle.predict_proba(x_test)
# print(f"ANN_ruc_auc_train  : {roc_auc_score(y_test, predict[:,1]):.2f}")
# print(train_model_by_kfold(df,ann_modle))


######################Tuning####################

# ####Hidden layer size graph####
#
# scaler = StandardScaler()
# scaler.fit(x_train)
# train_accs = []
# for size_ in range(1, 100, 2):
#    model = MLPClassifier(random_state=1,
#                          hidden_layer_sizes=(size_, size_),
#                          max_iter=100,
#                          activation='relu',
#                          verbose=False,
#                          learning_rate_init=0.001)
#    model.fit(scaler.transform(x_train), y_train)
#    train_acc = model.score(scaler.transform(x_train), y_train)
#    train_accs.append(train_acc)


#on validation  k-fold
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train =x_train.to_numpy()
# y_train=y_train.to_numpy()
# for train_idx, val_idx in kf.split(x_train):
#    test_accs = []
#    for size_ in range(1, 100, 2):
#        model = MLPClassifier(random_state=1,
#                              hidden_layer_sizes=(size_, size_),
#                              max_iter=100,
#                              activation='relu',
#                              verbose=False,
#                              learning_rate_init=0.001,
#                              alpha=0.00)
#        model.fit(scaler.transform(x_train[train_idx]), y_train[train_idx])
#        test_acc = model.score(scaler.transform(x_train[val_idx]), y_train[val_idx])
#        test_accs.append(test_acc)
#
# plt.figure(figsize=(7, 4))
# plt.plot(range(1, 100, 2), train_accs, label='Train')
# plt.plot(range(1, 100, 2), test_accs, label='Test')
# plt.legend()
# plt.show()
#
#
########### GridSerach############
# model = MLPClassifier(max_iter=10)
# parameter_space = {'hidden_layer_sizes':[(7,7),(17,17),(57,57),(97,97)],
#                    'activation':['tanh','relu','identity', 'logistic'],
#                    'solver':['sgd','adam','lbfgs'],
#                    'alpha':[0.0001,0.05],}
# clf = GridSearchCV(model,parameter_space,n_jobs=1,scoring='roc_auc',cv=10)
# clf.fit(x_train_s,y_train)
# print('Best parameters found:\n', clf.best_params_)
#
#
# ######## Tuning ANN ########
ann_model_ht = MLPClassifier(random_state=1,
                     hidden_layer_sizes=(97, 97),
                     activation='relu',
                     verbose=False,
                     solver='adam',
                     alpha=0.05,
                     )
ann_model_ht.fit(x_train, y_train)
# predict = ann_model_ht.predict_proba(x_test)
# print(f"ANN_ht_ruc_auc_train  : {roc_auc_score(y_test, predict[:,1]):.2f}")
# #
# train_model_by_kfold(df,ann_model_ht)


############# confusion matrix #############

ann_predict = ann_model_ht.predict(x_test)
ann_cm=confusion_matrix(y_test,ann_predict)
print("The confusion matrix for ANN model is: ")
print(ann_cm)

predictions = ann_model_ht.predict(x_test)
cm= confusion_matrix(y_test, predictions, labels=ann_model_ht.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ann_model_ht.classes_)
disp.plot()
plt.show()



########################## SVM ############################

# svm_model = SVC(probability=True)
# svm_model.fit(x_train, y_train)
# predictions = svm_model.predict_proba(x_test)
# print(f"Svm_roc_auc: {roc_auc_score(y_test, predictions[:,1]):.3f}")
#
# print(train_model_by_kfold(df,svm_model))

############# Tuning #####################
# model = SVC(random_state=1, probability=True)
# param_grid = {'C': [0.1, 1, 10, 100, 1000],
#              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#              'kernel': ['linear']}
#
# random_search = RandomizedSearchCV(model, param_distributions=param_grid, scoring='roc_auc', cv=10,
#                                   random_state=1, refit=True)
# random_search.fit(x_train, y_train)
# print(random_search.best_params_)


############# Tuning_svm ##################################
svm_model_ht = SVC(random_state=1, C=0.1, gamma=1, kernel='linear', probability=True)
svm_model_ht.fit(x_train, y_train)
# predictions = svm_model_ht.predict_proba(x_test)
# print(f"Svm_ht_roc_auc: {roc_auc_score(y_test, predictions[:,1]):.3f}")

# print(train_model_by_kfold(df,svm_model_ht))

################### linear equasion ######################

# from sklearn.svm import LinearSVC
# from sklearn.pipeline import make_pipeline
# from sklearn.datasets import make_classification
#
# clf = make_pipeline(StandardScaler(),
#                   LinearSVC(random_state=0, tol=1e-5))
# clf.fit(x_train, y_train)
# Pipeline(steps=[('standardscaler', StandardScaler()),
#                 ('linearsvc', LinearSVC(random_state=0, tol=1e-05))])
# print(clf.named_steps['linearsvc'].coef_)


################## clustering ##################

from tqdm import tqdm
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn_extra.cluster import KMedoids
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.metrics import silhouette_score, davies_bouldin_score


# kmedoids = KMedoids(random_state=0).fit(x_train)
# # #Sum of distances of samples to their closest cluster center.
# predict = kmedoids.predict(x_test)
# # print(predict)


# metric = distance_metric(type_metric.GOWER, max_range=x_train.max(axis=0))
# inerita_list = []
# sil_list = []
# dbi_list = []
#
# for num_clusters in tqdm(range(2,10,1)):
#     kmedoids = KMedoids(random_state=0,n_clusters=num_clusters, metric=metric).fit(x_train)
#     assignment = kmedoids.predict(x_test)
#     inerita_list.append(kmedoids.inertia_)
#     sil = silhouette_score(x_test, assignment)
#     sil_list.append(sil)
#     dbi = davies_bouldin_score(x_test, assignment)
#     dbi_list.append(dbi)
#     print(sil)
#
# plt.plot(range(2, 10, 1), inerita_list, marker='o')
# plt.title("Inertia")
# plt.xlabel("Number of clusters")
# plt.show()
#
# plt.plot(range(2, 10, 1), sil_list, marker='o')
# plt.title("Silhouette")
# plt.xlabel("Number of clusters")
# plt.show()
#
# plt.plot(range(2, 10, 1), dbi_list, marker='o')
# plt.title("Davies-bouldin")
# plt.xlabel("Number of clusters")
# plt.show()

############# Evaluation ##################

DT_predict = model_ht.predict(x_test)
DT_cm=confusion_matrix(y_test,DT_predict)
print("The confusion matrix for DT model is: ")
print(DT_cm)


SVM_predict = svm_model_ht.predict(x_test)
SVM_cm=confusion_matrix(y_test,SVM_predict)
print("The confusion matrix for svm model is: ")
print(SVM_cm)


SVM_predict = svm_model_ht.predict(x_test)
SVM_cm=confusion_matrix(y_test,SVM_predict)
print("The confusion matrix for svm model is: ")
print(SVM_cm)
# print(f"ANN__cm_Ruc_auc_score: {roc_auc_score(y_train,ann_model_ht.predict(x_train)):.3f}")
# print(confusion_matrix(y_true=y_train, y_pred=ann_model_ht.predict(x_train)))
#
# print(f"DT__cm_Ruc_auc_score: {roc_auc_score(y_train,model_ht.predict(x_train)):.3f}")
# print(confusion_matrix(y_true=y_train, y_pred=model_ht.predict(x_train)))
#
# print(f"SVM__cm_Ruc_auc_score: {roc_auc_score(y_train,svm_model_ht.predict(x_train)):.3f}")
# print(confusion_matrix(y_true=y_train, y_pred=svm_model_ht.predict(x_train)))


#################### improvment #######################

company_size = df_improve['company_size']
for i in company_size.keys():
    if company_size[i] == '10-49' or company_size[i] == '50-99':
        company_size[i] = 'small_company'
    else:
        company_size[i] = 'big_company'

df_improve_dummies = pd.DataFrame(df_improve,columns=['relevent_experience','company_size'])
df_with_dummies = pd.get_dummies(df_improve_dummies)
df_improve = pd.concat([df_improve, df_with_dummies], axis=1)
df_improve = df_improve.drop(columns=(['city','gender','enrollee_id','relevent_experience','major_discipline','company_size','company_type']))

x = df_improve.drop('target',axis=1)
y = df_improve['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
#
#
# param_grid = {'max_depth': np.arange(1, 10, 1),
#               'n_estimators': [100,200,300,400,500,600,700],
#               'criterion': ['entropy','gini'],
#               'max_features': ['auto', 'sqrt', 'log2', None],
#               'min_samples_leaf':np.arange(1, 20, 1),
#               'min_samples_split': np.arange(2,20,1)
#              }
# grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
#                            param_grid=param_grid,
#                            refit=True,
#                            cv=10,scoring='roc_auc')
#
# grid_search.fit(x_train, y_train)
# print(grid_search.best_params_)
# #
#
# rf_pipe = Pipeline(steps =[ ('std_scale',StandardScaler()), ("RF",RandomForestClassifier(criterion='gini',max_depth=5,max_features=None ,random_state=27, min_samples_leaf=8,min_samples_split= 2 ))])
# rf_pipe.fit(x_train,y_train)
# training_score = roc_auc_score(y_test, rf_pipe.predict_proba(x_test)[:,1])
# print(f"RF_ruc_auc:{training_score:.2f}")
#
# print(train_model_by_kfold(df_improve,rf_pipe))

##################### x_test ###########################

# td = pd.read_csv("C:\\Users\\Owner\\Downloads\\X_test.csv")
#
# #################################### cleaning the dataset ########################
# td['company_size'] = td['company_size'].replace('10/49', '10-49')
# td['experience'] = td['experience'].replace('>20', 21)
# td['experience'] = td['experience'].replace('<1', 0)
# td['last_new_job'] = td['last_new_job'].replace('>4',5)
# td['last_new_job'] = td['last_new_job'].replace('never',0)
#
# ################### add values to enrolled_university column with null ############################
#
# td['enrolled_university'].fillna('no_enrollment', inplace=True)
#
# ################### add values to edcuation level column with null ############################
#
# td['education_level'].fillna('High School', inplace=True)
#
# ################# add values to major discipline column with null ##############################
# for i in td['education_level']:
#     if i == 'Graduate' or i == 'Masters':
#         td['major_discipline'].fillna('STEM', inplace=True)
#     elif i == 'High School' or i == 'Primary School':
#         td['major_discipline'].fillna('No Major', inplace=True)
#
# ################### last new job null values #####################
# td['last_new_job'].fillna(int(1), inplace=True)
# td['last_new_job'] = td['last_new_job'].astype('int64')
# #print(df['last_new_job'].value_counts())
#
# ################# gender fill missing values ##################
# td['gender'].fillna('Male', inplace= True)
#
# ######################## remove experience with null values        ##################
# td = td.dropna(subset = ['experience'])
# # print(df['city_development_index'][0:10])
# #################################### fill company_size####################
# td['company_size'].fillna('50-99', inplace = True)
#
# ########################## fill company_type #######################
#
# td['company_type'].fillna('Pvt Ltd',inplace= True)
#
# #################### ordinal categories ################333
# td['education_level'] = td['education_level'].map({'Primary School':int(0),'Graduate':int(2) ,'High School':int(1),
#                                                                    'Phd':int(4),'Masters':int(3)})
# td['enrolled_university'] = td['enrolled_university'].map({'no_enrollment':int(0),'Part time course':int(1) ,'Full time course':int(2),})
#
# ##################### normalzation #######################
# td['experience'] = td['experience'].astype('float64')
# td['training_hours'] = (td['training_hours'] - min(td['training_hours'])) / (max(td['training_hours']) - min(td['training_hours']))
# td['experience'] = (td['experience'] - min(td['experience'])) / (max(td['experience']) - min(td['experience']))
#
# td_improve = pd.DataFrame(td)
# ################### dummies ###########################
# td_dummies = pd.DataFrame(td,columns=['relevent_experience','company_size','company_type'])
# new_td = pd.get_dummies(td_dummies)
# td = td.drop(columns=(['city','gender','enrollee_id','relevent_experience','major_discipline','company_size','company_type']))
# td = pd.concat([td, new_td], axis=1)
#
#
# company_size = td_improve['company_size']
# for i in company_size.keys():
#     if company_size[i] == '10-49' or company_size[i] == '50-99':
#         company_size[i] = 'small_company'
#     else:
#         company_size[i] = 'big_company'
#
# td_improve_dummies = pd.DataFrame(td_improve,columns=['relevent_experience','company_size'])
# td_with_dummies = pd.get_dummies(td_improve_dummies)
# td_improve = pd.concat([td_improve, td_with_dummies], axis=1)
# td_improve = td_improve.drop(columns=(['city','gender','enrollee_id','relevent_experience','major_discipline','company_size','company_type']))
# x_test = td_improve
# print(x_test.shape)
#
# ######################## y_test predict   ##################
#
# test_model = Pipeline(steps =[ ('std_scale',StandardScaler()), ("RF",RandomForestClassifier(criterion='gini',max_depth=5,max_features=None ,random_state=27, min_samples_leaf=8,min_samples_split= 2 ))])
# test_model.fit(x_train,y_train)
# y_test = test_model.predict(x_test)
# DF = pd.DataFrame(y_test)
# # save the dataframe as a csv file
# DF.to_csv("27_ytest.csv")