# Install Packages and Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from scipy import stats
from scipy.stats import chi2_contingency
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE
from imblearn.under_sampling import ClusterCentroids, AllKNN, RandomUnderSampler, CondensedNearestNeighbour
from imblearn.ensemble import EasyEnsembleClassifier, BalancedBaggingClassifier, RUSBoostClassifier, BalancedRandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, mean_squared_error, roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay 
from sklearn.feature_selection import RFE, chi2

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 1 - Data Prep (adding dummies, converting to parquet and loading data)
attrition_ind = pd.get_dummies(train['Attrition'])
train = pd.concat([train, attrition_ind], axis=1)
train = train.drop(['Left', 'Attrition'], axis='columns')
attrition_ind = pd.get_dummies(test['Attrition'])
test = pd.concat([test, attrition_ind], axis=1)
test = test.drop(['Left', 'Attrition'], axis='columns')

role_dummy = pd.get_dummies(train['Job Role'], prefix = 'role')
train = pd.concat([train, role_dummy], axis=1)
train = train.drop(['Job Role'], axis='columns')
role_dummy = pd.get_dummies(test['Job Role'], prefix = 'role')
test = pd.concat([test, role_dummy], axis=1)
test = test.drop(['Job Role'], axis='columns')

satisfaction_dummy = pd.get_dummies(train['Job Satisfaction'], prefix = 'satisfaction')
train = pd.concat([train, satisfaction_dummy], axis=1)
train = train.drop(['Job Satisfaction'], axis='columns')
satisfaction_dummy = pd.get_dummies(test['Job Satisfaction'], prefix = 'satisfaction')
test = pd.concat([test, satisfaction_dummy], axis=1)
test = test.drop(['Job Satisfaction'], axis='columns')

performance_dummy = pd.get_dummies(train['Performance Rating'], prefix = 'performance')
train = pd.concat([train, performance_dummy], axis=1)
train = train.drop(['Performance Rating'], axis='columns')
performance_dummy = pd.get_dummies(test['Performance Rating'], prefix = 'performance')
test = pd.concat([test, performance_dummy], axis=1)
test = test.drop(['Performance Rating'], axis='columns')

gender_dummy = pd.get_dummies(train['Gender'])
train = pd.concat([train, gender_dummy], axis=1)
train = train.drop(['Gender'], axis='columns')
gender_dummy = pd.get_dummies(test['Gender'])
test = pd.concat([test, gender_dummy], axis=1)
test = test.drop(['Gender'], axis='columns')

overtime_dummy = pd.get_dummies(train['Overtime'], prefix = 'overtime')
train = pd.concat([train, overtime_dummy], axis=1)
train = train.drop(['Overtime'], axis='columns')
overtime_dummy = pd.get_dummies(test['Overtime'], prefix = 'overtime')
test = pd.concat([test, overtime_dummy], axis=1)
test = test.drop(['Overtime'], axis='columns')

education_dummy = pd.get_dummies(train['Education Level'], prefix = 'education')
train = pd.concat([train, education_dummy], axis=1)
train = train.drop(['Education Level'], axis='columns')
education_dummy = pd.get_dummies(test['Education Level'], prefix = 'education')
test = pd.concat([test, education_dummy], axis=1)
test = test.drop(['Education Level'], axis='columns')

marital_dummy = pd.get_dummies(train['Marital Status'], prefix = 'marital')
train = pd.concat([train, marital_dummy], axis=1)
train = train.drop(['Marital Status'], axis='columns')
marital_dummy = pd.get_dummies(test['Marital Status'], prefix = 'marital')
test = pd.concat([test, marital_dummy], axis=1)
test = test.drop(['Marital Status'], axis='columns')

level_dummy = pd.get_dummies(train['Job Level'], prefix = 'level')
train = pd.concat([train, level_dummy], axis=1)
train = train.drop(['Job Level'], axis='columns')
level_dummy = pd.get_dummies(test['Job Level'], prefix = 'level')
test = pd.concat([test, level_dummy], axis=1)
test = test.drop(['Job Level'], axis='columns')

size_dummy = pd.get_dummies(train['Company Size'], prefix = 'size')
train = pd.concat([train, size_dummy], axis=1)
train = train.drop(['Company Size'], axis='columns')
size_dummy = pd.get_dummies(test['Company Size'], prefix = 'size')
test = pd.concat([test, size_dummy], axis=1)
test = test.drop(['Company Size'], axis='columns')

remote_dummy = pd.get_dummies(train['Remote Work'], prefix = 'remote')
train = pd.concat([train, remote_dummy], axis=1)
train = train.drop(['Remote Work'], axis='columns')
remote_dummy = pd.get_dummies(test['Remote Work'], prefix = 'remote')
test = pd.concat([test, remote_dummy], axis=1)
test = test.drop(['Remote Work'], axis='columns')

leadership_dummy = pd.get_dummies(train['Leadership Opportunities'], prefix = 'leadership')
train = pd.concat([train, leadership_dummy], axis=1)
train = train.drop(['Leadership Opportunities'], axis='columns')
leadership_dummy = pd.get_dummies(test['Leadership Opportunities'], prefix = 'leadership')
test = pd.concat([test, leadership_dummy], axis=1)
test = test.drop(['Leadership Opportunities'], axis='columns')

innovation_dummy = pd.get_dummies(train['Innovation Opportunities'], prefix = 'innovation')
train = pd.concat([train, innovation_dummy], axis=1)
train = train.drop(['Innovation Opportunities'], axis='columns')
innovation_dummy = pd.get_dummies(test['Innovation Opportunities'], prefix = 'innovation')
test = pd.concat([test, innovation_dummy], axis=1)
test = test.drop(['Innovation Opportunities'], axis='columns')

reputation_dummy = pd.get_dummies(train['Company Reputation'], prefix = 'reputation')
train = pd.concat([train, reputation_dummy], axis=1)
train = train.drop(['Company Reputation'], axis='columns')
reputation_dummy = pd.get_dummies(test['Company Reputation'], prefix = 'reputation')
test = pd.concat([test, reputation_dummy], axis=1)
test = test.drop(['Company Reputation'], axis='columns')

recognition_dummy = pd.get_dummies(train['Employee Recognition'], prefix = 'recognition')
train = pd.concat([train, recognition_dummy], axis=1)
train = train.drop(['Employee Recognition'], axis='columns')
recognition_dummy = pd.get_dummies(test['Employee Recognition'], prefix = 'recognition')
test = pd.concat([test, recognition_dummy], axis=1)
test = test.drop(['Employee Recognition'], axis='columns')

balance_dummy = pd.get_dummies(train['Work-Life Balance'], prefix = 'balance')
train = pd.concat([train, balance_dummy], axis=1)
train = train.drop(['Work-Life Balance'], axis='columns')
balance_dummy = pd.get_dummies(test['Work-Life Balance'], prefix = 'balance')
test = pd.concat([test, balance_dummy], axis=1)
test = test.drop(['Work-Life Balance'], axis='columns')

train = train.drop(['Employee ID'], axis='columns')
test = test.drop(['Employee ID'], axis='columns')

train['age_minus_company_tenure'] = train['Age'] - train['Company Tenure']
test['age_minus_company_tenure'] = test['Age'] - test['Company Tenure']

train['age_minus_years_at_company'] = train['Age'] - train['Years at Company']
test['age_minus_years_at_company'] = test['Age'] - test['Years at Company']

# Converting to parquet format for quicker processing
columns = train.columns
train.to_parquet('train.parquet.gzip',compression='gzip')
test.to_parquet('test.parquet.gzip',compression='gzip')

del train
del test

train = pd.read_parquet('train.parquet.gzip', columns=columns)
test = pd.read_parquet('test.parquet.gzip', columns=columns)

X_train = train[['role_Education', 'role_Finance', 'role_Healthcare', 'role_Media',
       'role_Technology', 'satisfaction_High', 'satisfaction_Low',
       'satisfaction_Medium', 'satisfaction_Very High', 'performance_Average',
       'performance_Below Average', 'performance_High', 'performance_Low',
       'Female', 'Male', 'overtime_No', 'overtime_Yes',
       'education_Associate Degree', "education_Bachelor’s Degree",
       'education_High School', "education_Master’s Degree", 'education_PhD',
       'marital_Divorced', 'marital_Married', 'marital_Single', 'level_Entry',
       'level_Mid', 'level_Senior', 'size_Large', 'size_Medium', 'size_Small',
       'remote_No', 'remote_Yes', 'leadership_No', 'leadership_Yes',
       'innovation_No', 'innovation_Yes', 'reputation_Excellent',
       'reputation_Fair', 'reputation_Good', 'reputation_Poor',
       'recognition_High', 'recognition_Low', 'recognition_Medium',
       'recognition_Very High', 'balance_Excellent', 'balance_Fair',
       'balance_Good', 'balance_Poor', 'Age', 'Years at Company', 'Monthly Income', 'Number of Promotions',
       'Distance from Home', 'Number of Dependents', 'Company Tenure', 'age_minus_company_tenure',
       'age_minus_years_at_company']]
y_train = train[['Stayed']]

X_train_binary = X_train[['role_Education', 'role_Finance', 'role_Healthcare', 'role_Media',
       'role_Technology', 'satisfaction_High', 'satisfaction_Low',
       'satisfaction_Medium', 'satisfaction_Very High', 'performance_Average',
       'performance_Below Average', 'performance_High', 'performance_Low',
       'Female', 'Male', 'overtime_No', 'overtime_Yes',
       'education_Associate Degree', "education_Bachelor’s Degree",
       'education_High School', "education_Master’s Degree", 'education_PhD',
       'marital_Divorced', 'marital_Married', 'marital_Single', 'level_Entry',
       'level_Mid', 'level_Senior', 'size_Large', 'size_Medium', 'size_Small',
       'remote_No', 'remote_Yes', 'leadership_No', 'leadership_Yes',
       'innovation_No', 'innovation_Yes', 'reputation_Excellent',
       'reputation_Fair', 'reputation_Good', 'reputation_Poor',
       'recognition_High', 'recognition_Low', 'recognition_Medium',
       'recognition_Very High', 'balance_Excellent', 'balance_Fair',
       'balance_Good', 'balance_Poor']]
X_train_continuous = X_train[['Age', 'Years at Company', 'Monthly Income', 'Number of Promotions',
       'Distance from Home', 'Number of Dependents', 'Company Tenure', 'age_minus_company_tenure',
       'age_minus_years_at_company']]

X_test_binary = test[['role_Education', 'role_Finance', 'role_Healthcare', 'role_Media',
       'role_Technology', 'satisfaction_High', 'satisfaction_Low',
       'satisfaction_Medium', 'satisfaction_Very High', 'performance_Average',
       'performance_Below Average', 'performance_High', 'performance_Low',
       'Female', 'Male', 'overtime_No', 'overtime_Yes',
       'education_Associate Degree', "education_Bachelor’s Degree",
       'education_High School', "education_Master’s Degree", 'education_PhD',
       'marital_Divorced', 'marital_Married', 'marital_Single', 'level_Entry',
       'level_Mid', 'level_Senior', 'size_Large', 'size_Medium', 'size_Small',
       'remote_No', 'remote_Yes', 'leadership_No', 'leadership_Yes',
       'innovation_No', 'innovation_Yes', 'reputation_Excellent',
       'reputation_Fair', 'reputation_Good', 'reputation_Poor',
       'recognition_High', 'recognition_Low', 'recognition_Medium',
       'recognition_Very High', 'balance_Excellent', 'balance_Fair',
       'balance_Good', 'balance_Poor']]
X_test_continuous = test[['Age', 'Years at Company', 'Monthly Income', 'Number of Promotions',
       'Distance from Home', 'Number of Dependents', 'Company Tenure', 'age_minus_company_tenure',
       'age_minus_years_at_company']]
y_test = test[['Stayed']]

chi_scores = chi2(X_train.drop('age_minus_company_tenure', axis=1), y_train)
chi_values = pd.Series(chi_scores[0], index=X_train.drop('age_minus_company_tenure', axis=1).columns)
chi_values.sort_values(ascending=False, inplace=True)
chi_values.plot.bar()
plt.savefig('Chi Square Plot')
plt.clf()

def chi2_test(feature):
    contingency_table = pd.crosstab(X_train.drop('age_minus_company_tenure', axis=1)[feature], train['Stayed'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return p

p_values = {feature: chi2_test(feature) for feature in X_train.drop('age_minus_company_tenure', axis=1).columns}
significant_features = {feature: p for feature, p in p_values.items() if p < 0.05}
print('Significant Features:', significant_features,  '\nTotal Significant Features:', len(significant_features))

# 2 - Feature Selection 

model = XGBClassifier(objective='binary:logistic', max_depth=5) # LogisticRegression()

rfe = RFE(model, n_features_to_select=50)
selector = rfe.fit(X_train, y_train)

removed_features = []

for i, col in zip(range(X_train.shape[1]), X_train.columns):

    if rfe.support_[i] == False:
        removed_features.append(col)

for i in removed_features:
    try:
        X_train_binary = X_train_binary.drop(i, axis=1)
        X_test_binary = X_test_binary.drop(i, axis=1)
    except:
        continue
    try:
        X_train_continuous = X_train_continuous.drop(i, axis=1)
        X_test_continuous = X_test_continuous.drop(i, axis=1)
    except:
        continue

# 3 - Feature Normalisation

gaussian_features = []
non_gaussian_features = []

for c in X_train_continuous.columns:
    data = list(X_train_continuous[c][:5000])
    normality_tests = {'Shapiro-Wilk Test': stats.shapiro(data)}
    
    for test_name, test_result in normality_tests.items():
        p_value = test_result[1]
        alpha = 0.05  # Significance level
        if p_value < alpha:
            non_gaussian_features.append(c)
        else:
            gaussian_features.append(c)

features = {'gaussian_features': gaussian_features, 'non_gaussian_features': non_gaussian_features}
np.save('features.npy', features) 

features = np.load('features.npy',allow_pickle='TRUE').item()

non_gaussian_features = features['non_gaussian_features']
gaussian_features = features['gaussian_features']

X_train_continuous_for_transformer = X_train_continuous[non_gaussian_features] # For Non-Gaussian Features
X_train_continuous_for_scaler = X_train_continuous[gaussian_features] # For Gaussian Features

X_test_continuous_for_transformer = X_test_continuous[non_gaussian_features] # For Non-Gaussian Features
X_test_continuous_for_scaler = X_test_continuous[gaussian_features] # For Gaussian Features

quantile_transformer = QuantileTransformer(n_quantiles=100, output_distribution='uniform')
quantile_transformer.fit(X_train_continuous_for_transformer)
X_train_continuous_for_transformer = quantile_transformer.transform(X_train_continuous_for_transformer)
X_train_continuous_for_transformer = pd.DataFrame(X_train_continuous_for_transformer, index=X_train_continuous.index, columns=non_gaussian_features)
X_test_continuous_for_transformer = quantile_transformer.transform(X_test_continuous_for_transformer)
X_test_continuous_for_transformer = pd.DataFrame(X_test_continuous_for_transformer, index=X_test_continuous.index, columns=non_gaussian_features)

scaler = StandardScaler()
if len(gaussian_features) > 0:
    scaler.fit(X_train_continuous_for_scaler)
    X_train_continuous_for_scaler = scaler.transform(X_train_continuous_for_scaler)
    X_train_continuous_for_scaler = pd.DataFrame(X_train_continuous_for_scaler, index=X_train_continuous.index, columns=gaussian_features)
    
    X_test_continuous_for_scaler = scaler.transform(X_test_continuous_for_scaler)
    X_test_continuous_for_scaler = pd.DataFrame(X_test_continuous_for_scaler, index=X_test_continuous.index, columns=gaussian_features)
else: 
    print('No Gaussian Features Found, Continuing to Model Training')

X_train = pd.concat([X_train_binary, X_train_continuous_for_scaler, X_train_continuous_for_transformer], axis=1)
X_test = pd.concat([X_test_binary, X_test_continuous_for_scaler, X_test_continuous_for_transformer], axis=1)

# 4 - Resampling (to even classes) them reporting results

resamplers = ['SMOTEENN', 'SMOTE', 'AllKNN', 'RandomUnderSampler']

roc_auc = []
mse = []
recall = []

for r in resamplers:
    print(f'\nTesting Resampler {r}')

    X_resampled, y_resampled = eval(r + "()").fit_resample(X_train, y_train)
    X_testing, X_review, y_testing, y_review = train_test_split(X_resampled, y_resampled, test_size=0.001, random_state=0)
    
    model = XGBClassifier(n_estimators=1000,
                            max_depth=6,
                            learning_rate=0.01,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            gamma=0,
                            reg_alpha=0.05,
                            reg_lambda=1,
                            scale_pos_weight=1,
                            objective='binary:logistic',
                            eval_metric='logloss')
    model.fit(X_testing, y_testing)

    predictions = model.predict(X_test)

    print('roc_auc_score: ', roc_auc_score(y_test, predictions))
    print('mean_squared_error: ', mean_squared_error(y_test, predictions))
    print('recall_score: ', recall_score(y_test, predictions))

    roc_auc.append(roc_auc_score(y_test, predictions))
    mse.append(mean_squared_error(y_test, predictions))
    recall.append(recall_score(y_test, predictions))

    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, predictions))
    disp.plot()
    plt.savefig(f'Confusion Matrix - {r} Sampler.png')
    plt.clf()

results = {'resampler': resamplers, 'roc_auc': roc_auc, 'mse': mse, 'recall': recall}

results = pd.DataFrame.from_dict(results)
results.to_csv('results.csv', index=False)  

