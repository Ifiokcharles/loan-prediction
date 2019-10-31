import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
#from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

'''Download Excel Data and store in dataframe'''
loan_data = pd.read_excel ('BankLoan_Data.xlsx',sep=',', header=0)
loan_data.drop("Loan_ID", axis=1, inplace=True)
loan_data.Dependents = loan_data.Dependents.replace({'3+':3})
describe_loan_data = loan_data.describe()

'''Calculating probabilties of getting loans based on several factors.'''
credit_history_prob = loan_data.pivot_table(values='Loan_Status',index=['Credit_History'],
                              aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

gender_prob = loan_data.pivot_table(values='Loan_Status',index=['Gender'],
                              aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

marital_prob = loan_data.pivot_table(values='Loan_Status',index=['Married'],
                              aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

education_prob = loan_data.pivot_table(values='Loan_Status',index=['Education'],
                              aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

self_employed_prob = loan_data.pivot_table(values='Loan_Status',index=['Self_Employed'],
                              aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

dependents_prob = loan_data.pivot_table(values='Loan_Status',index=['Dependents'],
                              aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

property_area_prob = loan_data.pivot_table(values='Loan_Status',index=['Property_Area'],
                              aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

loan_term_prob = loan_data.pivot_table(values='Loan_Status',index=['Loan_Amount_Term'],
                              aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

loan_term_prob['Loan_Term_Count'] = loan_data['Loan_Amount_Term'].value_counts()

'''Rearrange education column'''
inverse_education = [education_prob.Loan_Status[1], education_prob.Loan_Status[0]]

'''Plotting Bar charts to show relationships'''
N = len(credit_history_prob.index)
index = np.arange(N)
bar_width = 0.10

plt.figure()
plt.bar(index, credit_history_prob.Loan_Status, bar_width, alpha=1.0, color='red',label='Credit_History')
plt.bar(index + bar_width, marital_prob.Loan_Status, bar_width, alpha=0.9, color='firebrick', 
        label='Married')
plt.bar(index + (2*bar_width), inverse_education, bar_width, alpha=0.9, color='indianred', 
        label='Graduate')
plt.bar(index + (3*bar_width), self_employed_prob.Loan_Status, bar_width, alpha=0.9, 
        color='salmon', label='Self_Employed')
plt.ylabel('Propbability of Getting Loan', fontsize=12)
plt.title('Probability of getting a loan based on several factors', fontsize=15)
plt.xticks(index + (1.5*bar_width), ('No', 'Yes'))
plt.legend(loc='upper right', bbox_to_anchor=(1.16, 1), fontsize=10)
plt.tight_layout()

plt.figure()
plt.bar(gender_prob.index, gender_prob.Loan_Status, alpha=0.9, width = 0.10, color=['hotpink','royalblue'])
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Propbability of getting loan', fontsize=12)
plt.title('Probability of getting loan by gender', fontsize=15)

plt.figure()
plt.bar(dependents_prob.index, dependents_prob.Loan_Status, alpha=0.5, width = 0.13, color=['r','firebrick','indianred','salmon'])
plt.xlabel('Dependents', fontsize=12)
plt.xticks(dependents_prob.index, ('No child', '1 child', '2 children', '3+ children'))
plt.ylabel('Propbability of getting loan', fontsize=12)
plt.title('Probability of getting loan by dependents', fontsize=15)

plt.figure()
plt.barh(property_area_prob.index, property_area_prob.Loan_Status, alpha=0.5, height = 0.15, color=['r','firebrick','salmon'])
plt.ylabel('Property Area', fontsize=12)
plt.xlabel('Propbability of getting loan', fontsize=12)
plt.title('Probability of getting loan by property area', fontsize=15)

N = len(loan_term_prob.index)
index = np.arange(N)
bar_width = 0.10

plt.figure()
plt.bar(index, loan_term_prob.Loan_Status, alpha=0.9, color='orange')
plt.xlabel('Loan_Amount_Term', fontsize=12)
plt.ylabel('Propbability of getting loan', fontsize=12)
plt.xticks(index, ('12','36','60','84','120','180','240','300','360','480'))
plt.title('Probability of getting loan by loan_term', fontsize=15)

plt.figure()
plt.bar(index, loan_term_prob.Loan_Term_Count, alpha=0.9, color='orange')
plt.xlabel('Loan_Amount_Term', fontsize=12)
plt.ylabel('Sample size of Loan_Term', fontsize=12)
plt.xticks(index, ('12','36','60','84','120','180','240','300','360','480'))
plt.title('Sample size of Loan Term', fontsize=15)

'''Probability of getting loans when combining factors'''
credit_gender_prob = loan_data.pivot_table(values='Loan_Status',index=['Credit_History', 'Gender'],
                              aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

credit_married_prob = loan_data.pivot_table(values='Loan_Status',index=['Credit_History', 'Married'],
                              aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

credit_education_prob = loan_data.pivot_table(values='Loan_Status',index=['Credit_History', 'Education'],
                              aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

credit_employed_prob = loan_data.pivot_table(values='Loan_Status',index=['Credit_History', 'Self_Employed'],
                              aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

'''Plotting bar charts'''
loan_index0 = [credit_married_prob.Loan_Status[0][0],credit_education_prob.Loan_Status[0][1], 
               credit_employed_prob.Loan_Status[0][0]]

loan_index1 = [credit_married_prob.Loan_Status[0][1],credit_education_prob.Loan_Status[0][0], 
               credit_employed_prob.Loan_Status[0][1]]

loan_index2 = [credit_married_prob.Loan_Status[2],credit_education_prob.Loan_Status[3], 
               credit_employed_prob.Loan_Status[2]]

loan_index3 = [credit_married_prob.Loan_Status[3],credit_education_prob.Loan_Status[2], 
               credit_employed_prob.Loan_Status[3]]

N = len(loan_index0)
indi = np.arange(N)
bar_width=0.10
plt.figure()
plt.bar(indi, loan_index0, bar_width, alpha=0.9, color='crimson', label='No_credit_history, No')
plt.bar(indi+bar_width, loan_index1, bar_width, alpha=0.9, color='red', label='No_credit_history, Yes')
plt.bar(indi+(2*bar_width), loan_index2, bar_width, alpha=0.9, color='orange', label='Yes_credit_history, No')
plt.bar(indi+(3*bar_width), loan_index3, bar_width, alpha=1.0, color='gold', label='Yes_credit_history, Yes')
plt.xticks(indi + bar_width, ('CreditHistory_Married??', 'CreditHistory_Graduate??', 'CreditHistory_SelfEmployed??'), fontsize=10)
plt.title('Probability of getting loan by combination of factors', fontsize=15)
plt.ylabel('Probability of getting loan', fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), shadow=True, ncol=1, fontsize=13)
plt.tight_layout()

'''Box Plots'''
loan_data.boxplot(column='LoanAmount', by = ['Self_Employed','Education'])
plt.ylabel('Loan Amount')

loan_data.boxplot(column='LoanAmount', by = ['Self_Employed','Married'])
plt.ylabel('Loan Amount')

loan_data.boxplot(column='LoanAmount', by = ['Education','Married'])
plt.ylabel('Loan Amount')

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
import seaborn as sns; sns.set()
ax = sns.scatterplot(x="ApplicantIncome", y="LoanAmount", hue='Loan_Status', data=loan_data,
                     palette=dict(N="r", Y="b"))

'''Counting Data'''
count_self_employed = loan_data['Self_Employed'].value_counts()
count_married = loan_data['Married'].value_counts()

'''Filling missing values for Self Employed'''
loan_data['Self_Employed'].fillna('No', inplace=True)

'''Checking for missing values'''
missing_values = loan_data.apply(lambda x: sum(x.isnull()),axis=0) 

table = loan_data.pivot_table(values='LoanAmount', index='Education',
                              columns='Self_Employed', aggfunc=np.median)

def fill(x):
    if pd.isnull(x['LoanAmount']):
        return table.loc[x['Education']][x['Self_Employed']]
    else:
        return x['LoanAmount']

'''Replace missing values'''
loan_data['LoanAmount'].fillna(loan_data[loan_data['LoanAmount'].isnull()].apply(fill, axis=1), 
                               inplace=True)

'''Treat the extreme values in the distribution of LoanAmount and ApplicantIncome'''
loan_data['LoanAmount_log'] = np.log(loan_data['LoanAmount'])
plt.figure()
plt.hist(loan_data['LoanAmount_log'],bins=20, alpha=0.8)
plt.title('Histogram plot of the log of loan amount')

loan_data['TotalIncome'] = loan_data['ApplicantIncome'] + loan_data['CoapplicantIncome']
loan_data['TotalIncome_log'] = np.log(loan_data['TotalIncome'])
plt.figure()
plt.hist(loan_data['TotalIncome_log'],bins=20, color='maroon', alpha=0.8)
plt.title('Histogram plot of log of the total income')

'''Filling missing values'''
loan_data['Gender'].fillna(loan_data['Gender'].mode()[0], inplace=True)
loan_data['Married'].fillna(loan_data['Married'].mode()[0], inplace=True)
loan_data['Dependents'].fillna(loan_data['Dependents'].mode()[0], inplace=True)
loan_data['Loan_Amount_Term'].fillna(loan_data['Loan_Amount_Term'].mode()[0], inplace=True)
loan_data['Credit_History'].fillna(loan_data['Credit_History'].mode()[0], inplace=True)


from sklearn.preprocessing import LabelEncoder
var_mode = ['Gender','Married','Dependents', 'Education','Self_Employed','Property_Area','Loan_Status']
num_loan_data = LabelEncoder()
for i in var_mode:
    loan_data[i] = num_loan_data.fit_transform(loan_data[i])
            
'''Creating Models and looking for a champion model'''
predictors = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Loan_Amount_Term', 
              'Credit_History', 'Property_Area','LoanAmount_log','TotalIncome_log']

outcome = 'Loan_Status'

from matplotlib import cm as cm
corr_data = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Loan_Amount_Term', 
              'Credit_History', 'Property_Area','LoanAmount_log','TotalIncome_log', 'Loan_Status']

cmap = cm.get_cmap('coolwarm', 30)
f = plt.figure(figsize=(19, 15))
plt.matshow(loan_data[corr_data].corr(), fignum=f.number, cmap=cmap)
plt.xticks(range(loan_data[corr_data].shape[1]), loan_data[corr_data].columns, fontsize=8, rotation=50)
plt.yticks(range(loan_data[corr_data].shape[1]), loan_data[corr_data].columns, fontsize=12)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=12)
plt.tight_layout()

'''KMeans Clustering'''
from sklearn.cluster import KMeans
elbow = []
for i in range(1, 11):
    kmeans=KMeans(n_clusters = i, algorithm = 'auto', init='k-means++', n_init=10, verbose=0)
    kmeans.fit(loan_data[predictors])
    elbow.append(kmeans.inertia_)
plt.plot(range(1,11), elbow)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Elbow")
plt.show()

k = 5
loan_kmeans = KMeans(n_clusters = k, algorithm = 'auto', init='k-means++', n_init=10, verbose=0)
loan_kmeans.fit(loan_data[predictors])
loan_kmeans_predict = loan_kmeans.predict(loan_data[predictors])

from sklearn.decomposition import PCA
loan_pca = PCA(n_components=2)
loan_components = loan_pca.fit_transform(loan_data[predictors])

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Loan Component 1', fontsize = 10)
ax.set_ylabel('Loan Component 2', fontsize = 10)
ax.set_title('2loan component PCA', fontsize = 15)
colors = ['b','r','y','g','orange']
for color, i in zip(colors, [0,1,2,3,4]):
    ax.scatter(loan_components[loan_kmeans_predict == i, 0], loan_components[loan_kmeans_predict == i, 1], alpha=.8, color=color,
               s = 10)
ax.legend(loc='best', shadow=False, scatterpoints=1)

'''Train_Test_Split'''
from sklearn.model_selection import train_test_split
Loan_train, Loan_test, loan_train, loan_test = train_test_split(loan_data[predictors], 
                                                    loan_data[outcome],test_size = 0.30)

'''USING RANDOM FOREST CLASSIFIER MODEL'''
'''All attributes'''
classifier_rf_all = RandomForestClassifier(n_estimators=25, min_samples_split=25,
                                           max_depth=7, max_features=1)
classifier_rf_all.fit(Loan_train,loan_train)
prediction_rf_all = classifier_rf_all.predict(Loan_test)
'''Accuracies and Cross validation for All Attributes'''
accuracies_rf_all = cross_val_score(classifier_rf_all, loan_data[predictors], loan_data[outcome], cv=10)
accuracy_mean_rf_all = accuracies_rf_all.mean()

'''Important attributes'''
important_rf = pd.Series(classifier_rf_all.feature_importances_, 
                         index=predictors).sort_values(ascending=False)
important_pred_rf = [important_rf.index[0],important_rf.index[1],important_rf.index[2]]
classifier_rf = RandomForestClassifier(n_estimators=25, min_samples_split=25,
                                       max_depth=7, max_features=1)
Loan_train_imp, Loan_test_imp, loan_train_imp, loan_test_imp = train_test_split(loan_data[important_pred_rf], 
                                                    loan_data[outcome],test_size = 0.30)
classifier_rf.fit(Loan_train_imp,loan_train_imp)
prediction_rf_imp = classifier_rf.predict(Loan_test_imp)
'''Accuracies and Cross validation for Important Attributes'''
accuracies_rf_imp = cross_val_score(classifier_rf, loan_data[important_pred_rf], loan_data[outcome], cv=10)
accuracy_mean_rf_imp = accuracies_rf_imp.mean()
'''Confusion Matrix'''
matrix_rf = classification_report(loan_test_imp, prediction_rf_imp)
matrix_rf_conf = confusion_matrix(loan_test_imp, prediction_rf_imp)

'''OTHER MODELS'''
important_feat = ['Credit_History','TotalIncome_log','LoanAmount_log']
Loan_train_feat, Loan_test_feat, loan_train_feat, loan_test_feat = train_test_split(loan_data[important_feat], 
                                                    loan_data[outcome],test_size = 0.30)
'''USING SVC MODEL'''
'''All Attributes'''
classifier_svc_all = SVC(kernel='rbf', C=10, gamma=0.1)
classifier_svc_all.fit(Loan_train,loan_train)
prediction_svc_all = classifier_svc_all.predict(Loan_test)
'''Accuracies and Cross validation for all attributes'''
accuracies_svc_all = cross_val_score(classifier_svc_all, loan_data[predictors], loan_data[outcome], cv=10)
accuracy_mean_svc_all = accuracies_svc_all.mean()
'''Important Attributes'''
classifier_svc_imp = SVC(kernel='rbf', C=10, gamma=0.1)
classifier_svc_imp.fit(Loan_train_feat,loan_train_feat)
prediction_svc_imp = classifier_svc_imp.predict(Loan_test_feat)
'''Accuracies and Cross validation for important attributes'''
accuracies_svc_imp = cross_val_score(classifier_svc_imp, loan_data[important_feat], loan_data[outcome], cv=10)
accuracy_mean_svc_imp = accuracies_svc_imp.mean()
'''Confusion Matrix'''
matrix_svc = classification_report(loan_test_feat, prediction_svc_imp)
matrix_svc_conf = confusion_matrix(loan_test_feat, prediction_svc_imp)

'''USING DECISION TREE MODEL'''
'''All Attributes'''
classifier_tree_all = tree.DecisionTreeClassifier()
classifier_tree_all.fit(Loan_train,loan_train)
prediction_tree_all = classifier_tree_all.predict(Loan_test)
'''Accuracies and Cross validation for All Attributes'''
accuracies_tree_all = cross_val_score(classifier_tree_all, loan_data[predictors], loan_data[outcome], cv=10)
accuracy_mean_tree_all = accuracies_tree_all.mean()
'''Important Attributes'''
classifier_tree_imp = tree.DecisionTreeClassifier()
classifier_tree_imp.fit(Loan_train_feat,loan_train_feat)
prediction_tree_imp = classifier_tree_imp.predict(Loan_test_feat)
'''Accuracies and Cross validation for important attributes'''
accuracies_tree_imp = cross_val_score(classifier_tree_imp, loan_data[important_feat], loan_data[outcome], cv=10)
accuracy_mean_tree_imp = accuracies_tree_imp.mean()
'''Confusion Matrix'''
matrix_tree = classification_report(loan_test_feat, prediction_tree_imp)
matrix_tree_conf = confusion_matrix(loan_test_feat, prediction_tree_imp)

'''USING KNEIGHBORS'''
'''All Attributes'''
classifier_kn_all =  KNeighborsClassifier(n_neighbors=3)
classifier_kn_all.fit(Loan_train,loan_train)
prediction_kn_all = classifier_kn_all.predict(Loan_test)
'''Accuracies and Cross validation for All Attributes'''
accuracies_kn_all = cross_val_score(classifier_kn_all, loan_data[predictors], loan_data[outcome], cv=10)
accuracy_mean_kn_all = accuracies_kn_all.mean()
'''Important Attributes'''
classifier_kn_imp =  KNeighborsClassifier(n_neighbors=3)
classifier_kn_imp.fit(Loan_train_feat,loan_train_feat)
prediction_kn_imp = classifier_kn_imp.predict(Loan_test_feat)
'''Accuracies and Cross validation for important attributes'''
accuracies_kn_imp = cross_val_score(classifier_kn_imp, loan_data[important_feat], loan_data[outcome], cv=10)
accuracy_mean_kn_imp = accuracies_kn_imp.mean()
'''Confusion Matrix'''
matrix_kn = classification_report(loan_test_feat, prediction_kn_imp)
matrix_kn_conf = confusion_matrix(loan_test_feat, prediction_kn_imp)

'''Plotting Important Features Using Random Forest Classifier'''
n = [0,1,2,3,4,5,6,7,8,9] 
plt.figure()
plt.barh(n, important_rf, alpha=0.9, height = 0.50, color=['maroon'])
plt.ylabel('Features', fontsize=12)
plt.xlabel('Degree of Importance', fontsize=12)
plt.yticks(n,(important_rf.index))
plt.title('Relative Importance of all features', fontsize=15)

print(important_rf.index)

'''Plotting Cross Validations'''
n = [0,1,2,3,4,5,6,7,8,9]        
plt.figure()
plt.plot(n, accuracies_rf_imp, 'r-', marker = 'v', label = 'random_forest_IMP_features')
plt.plot(n, accuracies_svc_imp, 'gold', marker = 's', label = 'svm_IMP_features')
plt.plot(n, accuracies_tree_imp, 'g-', marker = 'o', label = 'tree_IMP_features')
plt.plot(n, accuracies_kn_imp, 'b-', marker = '*', label = 'knn_IMP_feature')
plt.xlabel('Ten fold')
plt.ylabel('Accuracy')
plt.legend(loc='upper left', bbox_to_anchor=(0.55, 0.16), shadow=True, ncol=2)
plt.title('Ten fold Validations of all algorithms using important features')
plt.grid()

plt.figure()
plt.plot(n, accuracies_rf_all, 'r-', marker = 'v', label = 'random_forest_ALL_features')
plt.plot(n, accuracies_svc_all, 'gold', marker = 's', label = 'svm_ALL_features')
plt.plot(n, accuracies_tree_all, 'g-', marker = 'o', label = 'tree_ALL_features')
plt.plot(n, accuracies_kn_all, 'b-', marker = '*', label = 'knn_ALL_feature')
plt.xlabel('Ten fold')
plt.ylabel('Accuracy')
plt.legend(loc='upper left', bbox_to_anchor=(0.25, 1), shadow=True, ncol=1)
plt.title('Ten fold Validations of all algorithms using all features')
plt.grid()

plt.figure()
plt.plot(n, accuracies_rf_imp, 'r-', marker = 'o', label = 'random_forest_IMP_features')
plt.plot(n, accuracies_rf_all, 'b-', marker = 's', label = 'random_forest_ALL_features')
plt.xlabel('Ten fold')
plt.ylabel('Accuracy')
plt.legend(loc='upper left', bbox_to_anchor=(0.8, 0.16), shadow=True, ncol=1)
plt.title('Comparing accuracies when using all attributes vs important features')
plt.grid()

plt.figure()
plt.plot(n, accuracies_svc_imp, 'r-', marker = 'o', label = 'svm_IMP_features')
plt.plot(n, accuracies_svc_all, 'b-', marker = 's', label = 'svm_ALL_features')
plt.xlabel('Ten fold')
plt.ylabel('Accuracy')
plt.legend(loc='upper left', bbox_to_anchor=(0.8, 0.16), shadow=True, ncol=1)
plt.title('Comparing accuracies when using all attributes vs important features')
plt.grid()

plt.figure()
plt.plot(n, accuracies_tree_imp, 'r-', marker = 'o', label = 'tree_IMP_features')
plt.plot(n, accuracies_tree_all, 'b-', marker = 's', label = 'tree_ALL_features')
plt.xlabel('Ten fold')
plt.ylabel('Accuracy')
plt.legend(loc='upper left', bbox_to_anchor=(0.8, 0.2), shadow=True, ncol=1)
plt.title('Comparing accuracies when using all attributes vs important features')
plt.grid()

plt.figure()
plt.plot(n, accuracies_kn_imp, 'r-', marker = 'o', label = 'knn_IMP_feature')
plt.plot(n, accuracies_kn_all, 'b-', marker = 's', label = 'knn_ALL_feature')
plt.xlabel('Ten fold')
plt.ylabel('Accuracy')
plt.legend(loc='upper left', bbox_to_anchor=(0.8, 0.9), shadow=True, ncol=1)
plt.title('Comparing accuracies when using all attributes vs important features')
plt.grid()

'''KMeans Cluster using top three features'''
kmeans_data = [ 'Credit_History','LoanAmount_log','TotalIncome_log']

k = 2
loan_kmeans = KMeans(n_clusters = k, algorithm = 'auto', init='k-means++', n_init=10, verbose=0)
loan_kmeans.fit(loan_data[kmeans_data])
loan_kmeans_predict = loan_kmeans.predict(loan_data[kmeans_data])

from sklearn.decomposition import PCA
loan_pca = PCA(n_components=2)
loan_components = loan_pca.fit_transform(loan_data[kmeans_data])

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Loan Component 1', fontsize = 10)
ax.set_ylabel('Loan Component 2', fontsize = 10)
ax.set_title('2loan component PCA', fontsize = 15)
colors = ['b','r']
for color, i in zip(colors, [0,1]):
    ax.scatter(loan_components[loan_kmeans_predict == i, 0], loan_components[loan_kmeans_predict == i, 1], alpha=.8, color=color,
               s = 10)
ax.legend(loc='best', shadow=False, scatterpoints=1)
ax.grid()
