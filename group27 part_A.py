import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import csv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import decomposition

df = pd.read_csv("C:\\Users\\micha\\Downloads\\XY_train.csv")

#################################### cleaning the dataset ########################
df['company_size'] = df['company_size'].replace('10/49', '10-49')
df['experience'] = df['experience'].replace('>20', 21)
df['experience'] = df['experience'].replace('<1', 0)
df['last_new_job'] = df['last_new_job'].replace('>4',5)
df['last_new_job'] = df['last_new_job'].replace('never',0)

##-----------------------------------------------------EDA---------------------------------------------------------
##-------------------------------------------continuous_variables--------------------------------------------------

###Boxplot city development
# plt.boxplot(df['city_development_index'], showfliers=False, showmeans=True)
# plt.title("Boxplot for city development", fontsize=20)
# plt.show()

###data and histogram for training hours
# print("min: ", min(df['training_hours']))
# print("max: ", max(df['training_hours']))
#
# print(df['training_hours'].mean())
# print(df['training_hours'].std())
# print(df['training_hours'].median())
# df['training_hours'].plot.hist(stacked=True, bins=20, color = 'lightblue')
# plt.title("Histogram for training hours", fontsize=20)
# plt.show()


###experience
# plt.figure(figsize=(10,6))
# sns.distplot(df['experience'],bins=15,color='#FF5733')
# plt.title('experience distribution')
# plt.show()

# ##---------------------------------Categorial variables---------------------------------------------------------

###city
# Source = Counter(df['city']).most_common(10)  ##select the top 10 common cities
# list1, list2 = zip(*Source)
# plt.bar(x=list1, height=list2, color=('lightblue'))
# plt.title("The 10 most common cities")
# plt.xlabel('Common city')
# plt.ylabel('Count')
# plt.xticks(rotation='vertical')
# plt.show()

###relevant experience
# labels, values = zip(*Counter(df['relevent_experience']).items())
# indexes = np.arange(len(labels))
# width = 1
# plt.bar(indexes, values, width = 0.7,color =('purple'))
# plt.title("relevent_experience")
# plt.xticks(indexes , labels)
# plt.show()

###enrolled university
# plot_gender = df['enrolled_university'].value_counts().reset_index()
# plot_gender.columns = ['enrolled_university','count']
# fig = px.pie(plot_gender,values='count',names='enrolled_university',template='simple_white',title='enrolled_university')
# fig.show()

###last new job

# labels = df.last_new_job.value_counts().keys()
# sizes = df.last_new_job.value_counts()
#
# colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#f0f8ff', '#E38C79']
#
# explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
#
# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
#         startangle=10, pctdistance=0.85, explode=explode,
#         labeldistance=1.1)
#
# centre_circle = plt.Circle((0, 0), 0.75, fc='white')
# fig = plt.gcf()
# fig.gca().add_artist(centre_circle)
#
# ax1.axis('equal')
# plt.tight_layout()
# plt.title('last new job')
# plt.show()

###target
# sns.countplot(data=df, x='target')
# plt.show()

### company size
# plot_gender = df['company_size'].value_counts().reset_index()
# plot_gender.columns = ['company_size','count']
#
# fig2 = px.pie(plot_gender,values='count',names='company_size',template='plotly_white',title='company_size')
# fig2.show()

###Company type
# labels, values =  zip(*Counter(df['company_type'].dropna()).items())
# indexes = np.arange(len(labels))
# width = 1
# plt.bar(indexes, values, width = 0.7,color =('red'))
# plt.title("company_type")
# plt.xticks(indexes , labels)
# plt.show()


###gender
# labels, values = zip(*Counter(df['gender'].dropna()).items())
# indexes = np.arange(len(labels))
# width = 1
# plt.bar(indexes, values, width = 0.7,color =('pink'))
# plt.xticks(indexes , labels)
# plt.title('gender')
# plt.show()

###Education level
# x = df[df['education_level'].notna()]
# res = x['education_level'].value_counts()
# res.plot(kind='pie', title='education level', autopct='%1.1f%%', legend=True)
# plt.show()

###major discipline
# plt.title('major_discipline')
# sns.countplot(data=df,x='major_discipline', hue='major_discipline')
# plt.show()

#######check if we have duplicates in our data##########
rows = len(df)
if df.duplicated().any():
    x = df.drop_duplicates(subset=None, keep="first", inplace=False)
    print("There were Duplicates. Duplicate samples are gone now")
    print("Dataset current length is: ", len(x))

else:
    print("No Duplicates")


###########check the missing values in every column in our data#############
def missing_values_check(df):
    MissingValue = df.isnull().sum()
    MissingValuePercent = (
                                  df.isnull().sum() / rows) * 100

    MissingValueTable = pd.concat([MissingValue, MissingValuePercent], axis=1)
    MissingValueTable = MissingValueTable.rename(
        columns={0: "Missing Values", 1: "% of Total Values"})
    # Create table with results

    MissingValueTable = MissingValueTable[
        MissingValueTable.iloc[:, 1] != 0].sort_values("% of Total Values").round(2)
    # Sort the table , ignore from columns without missing values

    return MissingValueTable


print(missing_values_check(df))



########## barplot for missing values in percentage
# missing_value = 100*df.isnull().sum()/len(df)
# missing_value = missing_value.reset_index()
# fig = px.bar(missing_value, y='missing values in percentage',x='variables',title='Missing values % in each column',
#              template='ggplot2')
# fig.show()


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
print(df['last_new_job'].value_counts())

################# gender fill missing values ##################
df['gender'].fillna('Male', inplace= True)

######################## remove experience with null values        ##################
df = df.dropna(subset = ['experience'])
print(df['city_development_index'][0:10])

################### missing null values in company type and company size with the knn module ####################

#label encoding for our categories features
Data1 = df.iloc[:,:].values
var = LabelEncoder()
Data1[:,1] = var.fit_transform(Data1[:,1])
Data1[:,3] =  var.fit_transform(Data1[:,3])
Data1[:,4] =  var.fit_transform(Data1[:,4])
Data1[:,5] =  var.fit_transform(Data1[:,5])
Data1[:,5] =  var.fit_transform(Data1[:,5])
Data1[:,6] =  var.fit_transform(Data1[:,6])
Data1[:,7] =  var.fit_transform(Data1[:,7])
Data1[:,9] =  var.fit_transform(Data1[:,9])
Data1[:,10] =  var.fit_transform(Data1[:,10])

df = pd.DataFrame(Data1,columns=['enrollee_id','city','city_development_index','gender','relevent_experience','enrolled_university','education_level','major_discipline','experience','company_size','company_type','last_new_job','training_hours','target'])
df2 = pd.DataFrame(Data1,columns=['enrollee_id','city','city_development_index','gender','relevent_experience','enrolled_university','education_level','major_discipline','experience','company_size','company_type','last_new_job','training_hours','target'])


df['company_size'] = df['company_size'].replace({8:np.nan}).astype('float64')
df2['company_type'] = df2['company_type'].replace({6:np.nan}).astype('float64')

df = df.drop(['company_type'],axis=1)
df2 = df2.drop(['company_size'], axis=1)
df_not_null = df.dropna()

y_train = df_not_null['company_size']
x_train = df_not_null.drop(['company_size'], axis =1)

#knn to fill missing values in company size
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)
x_test = df[df['company_size'].isnull()]
x_test = x_test.drop(['company_size'], axis =1)
y_pred = classifier.predict(x_test)
#print(y_pred)
#df = x_test+column("y_pred)
x_test['company_size'] = y_pred
# nulll in col company size in replace with y_pred

df['company_size'].fillna(x_test['company_size'], inplace=True)

#knn to fill missing values in company type
df2_not_null = df2.dropna()
y2_train = df2_not_null['company_type']
x2_train = df2_not_null.drop(['company_type'],axis = 1)
classifier2 = KNeighborsClassifier(n_neighbors=5)
classifier2.fit(x2_train,y2_train)
x2_test = df2[df2['company_type'].isnull()]
x2_test = x2_test.drop(['company_type'], axis=1)
y2_pred = classifier2.predict(x2_test)
#print(y2_pred)

x2_test['company_type'] = y2_pred
# nulll in col company size in replace with y_pred

df2['company_type'].fillna(x2_test['company_type'], inplace=True)
idx = 10
df.insert(loc = idx, column='company_type', value =df2['company_type'])
##-----------------------discritization for training hours-------
df['city_development_index'] = pd.qcut(df['city_development_index'], q=3, labels=('Low developed','Med developed','High developed'))

##-----------------------------feature extraction--------------------

############ create tech experience feature #####################

experience = df['experience']
major = df['major_discipline']
tech_experience = []
for i in range(len(experience)):
        if int(experience[i]) >= 8 and int(major[i]) == 5:
            tech_experience.append('Has tech experience')
        else:
            tech_experience.append('Dont have tech experience')
idx = 13
df.insert(loc = idx, column='Tech experience', value =tech_experience)

#############create quality_rate feature ###############

quality_count = 0
development = df['city_development_index']
education = df['education_level']
quality_rate = []
for i in range(len(development)):
    if development[i] == 'High developed':
        quality_count  +=1
    if development[i] == 'Med developed':
        quality_count +=0.5
    if development[i] == 'Low developed':
        quality_count += 0
    if education[i] == 4 or education[i] == 1:
        quality_count += 0.5
    if education[i] == 0 :
        quality_count += 1
    if education[i] == 2:
        quality_count += 1.5
    if education[i] == 3:
        quality_count +=2
    quality_rate.append(quality_count)
    quality_count = 0

idx = 14
df.insert(loc = idx, column='quality rate', value =quality_rate)

##----------------------Feature Representation ------------------

########### training hours normalization######################
df['training_hours'] = (df['training_hours'] - min(df['training_hours'])) / (max(df['training_hours']) - min(df['training_hours']))


# -----------------------------------Feature selection-----------------------

df['Tech experience'] = df['Tech experience'].replace('Has tech experience', 1)
df['Tech experience'] = df['Tech experience'].replace('Dont have tech experience', 0)

#######check the connection between target and our variabels#####

###cor with last new job
# corr1 = sns.catplot(y="target",x="last_new_job",data=df, kind="bar")
# corr1.fig.suptitle('correlation between last new job and target')
# plt.show()

### cor with quality rate
# corr2 = sns.lineplot(x='quality rate', y='target',data=df)
# corr2.set(title = 'correlation between quality rate and target')
# plt.show()

### corr with enrolled university
# corr3 = sns.lineplot(x='enrolled_university', y='target', palette='Set2', data=df)
# corr3.set(title = 'correlation between enrolled_university and target')
# plt.show()

###corr with education level
# df['education_level'] = df['education_level'].map({0:int(2),2:int(3) ,1:int(1),
#                                                                    3:int(4),4:int(0)})
# corr4 = sns.lineplot(x='education_level', y='target', palette='Set2', data=df)
# corr4.set(title = 'correlation between education level and target')
# plt.show()


###corr with company size
# corr5 = sns.lineplot(x='company_size', y='target', palette='Set2', data=df)
# corr5.set(title = 'correlation between company size level and target')
# plt.show()

###corr with company type
# corr6 = sns.lineplot(x='company_type', y='target', palette='Set2', data=df)
# corr6.set(title = 'correlation between company type level and target')
# plt.show()

###corr with training hours
# corr7 = sns.lineplot(x='training_hours', y='target', palette='Set2', data=df)
# corr7.set(title = 'correlation between training hours level and target')
# plt.show()

###corr with tech experience
# corr8 = sns.catplot(x='Tech experience',hue='target',data=df,kind="count")
# plt.show()

##--------------------corellation metrix--------------

#change the type so we can use the data
df['training_hours'] = df['training_hours'].astype('float64')
df['experience'] = df['experience'].astype('float64')
df['company_size']= df['company_size'].astype('category')
df['company_type']= df['company_type'].astype('category')
df['Tech experience']= df['Tech experience'].astype('float64')
df['last_new_job']= df['last_new_job'].astype('float64')
df['relevent_experience'] = df['relevent_experience'].astype('float64')
df['target']= df['target'].astype('float64')

# corr_mat_with_new_features = pd.DataFrame(df, columns=['target','training_hours', 'last_new_job','experience','quality rate'])
# ax = plt.axes()
# sns.heatmap(corr_mat_with_new_features.corr(), annot=True, cmap='RdYlBu')
# ax.set_title('Correlation Matrix')
# plt.show()

# #-----------------------------------Wrappers- stepwise regression --------------------------------------------------

# step 1 :
def stepwise_regression():
    results = sm.OLS(df['target'], df[['experience','Tech experience','relevent_experience','quality rate']]).fit()
    print(results.summary())

stepwise_regression()

# step 2 :
def stepwise_regression():
    results = sm.OLS(df['target'], df[['Tech experience','relevent_experience','quality rate']]).fit()
    print(results.summary())

stepwise_regression()

######## chi2 test for categorial variables########
from scipy.stats import chi2_contingency

#company_Size
# info = pd.crosstab(df['target'],df['company_size'])
# result = (chi2_contingency(info))
# significance_level = 0.05
# print("p value: " + str(result[1]))
# if result[1] <= significance_level:
#      print('Reject NULL HYPOTHESIS')
# else:
#      print('ACCEPT NULL HYPOTHESIS')
#
# #company type
# info = pd.crosstab(df['target'],df['company_type'])
# result = (chi2_contingency(info))
# significance_level = 0.05
# print("p value: " + str(result[1]))
# if result[1] <= significance_level:
#      print('Reject NULL HYPOTHESIS')
# else:
#      print('ACCEPT NULL HYPOTHESIS')
#
# #enrolled_university
# info = pd.crosstab(df['target'],df['enrolled_university'])
# result = (chi2_contingency(info))
# significance_level = 0.05
# print("p value: " + str(result[1]))
# if result[1] <= significance_level:
#      print('Reject NULL HYPOTHESIS')
# else:
#      print('ACCEPT NULL HYPOTHESIS')
#
#
# #education level
#
# info = pd.crosstab(df['target'],df['education_level'])
# result = (chi2_contingency(info))
# significance_level = 0.05
# print("p value: " + str(result[1]))
# if result[1] <= significance_level:
#      print('Reject NULL HYPOTHESIS')
# else:
#      print('ACCEPT NULL HYPOTHESIS')

##--------------------------Dimensionality reduction-----------------------------

###PCA
# PCA_data = df.drop(columns=(['city','city_development_index','gender','experience','last_new_job','training_hours','target','enrollee_id','major_discipline']))
# PCA_data = scale(PCA_data)
# pca = decomposition.PCA(n_components = 7)
# pca.fit(PCA_data)
# scores = pca.transform(PCA_data)
# scores_df = pd.DataFrame(scores,columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7'])
# print(scores_df)
#
# explained_varience = pca.explained_variance_ratio_
# print(explained_varience)
# per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
# labels = ['PC'+str(x) for x in range(1, len(per_var)+1)]
# plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
# plt.ylabel('Percentage of Explained Variance')
# plt.xlabel('Principal Component')
# plt.title('Scree Plot')
# plt.show()