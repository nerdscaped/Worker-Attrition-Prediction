# Install Packages and Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('train.csv')
print(data.columns) # Columns included ['Employee ID', 'Age', 'Gender', 'Years at Company', 'Job Role',
    #    'Monthly Income', 'Work-Life Balance', 'Job Satisfaction',
    #    'Performance Rating', 'Number of Promotions', 'Overtime',
    #    'Distance from Home', 'Education Level', 'Marital Status',
    #    'Number of Dependents', 'Job Level', 'Company Size', 'Company Tenure',
    #    'Remote Work', 'Leadership Opportunities', 'Innovation Opportunities',
    #    'Company Reputation', 'Employee Recognition', 'Attrition']

# Reviewing Data - using loop to go through columns 
columns = ['Employee ID', 'Age', 'Gender', 'Years at Company', 'Job Role',
       'Monthly Income', 'Work-Life Balance', 'Job Satisfaction',
       'Performance Rating', 'Number of Promotions', 'Overtime',
       'Distance from Home', 'Education Level', 'Marital Status',
       'Number of Dependents', 'Job Level', 'Company Size', 'Company Tenure',
       'Remote Work', 'Leadership Opportunities', 'Innovation Opportunities',
       'Company Reputation', 'Employee Recognition']

attrition_ind = pd.get_dummies(data['Attrition'])
data = pd.concat([data, attrition_ind], axis=1)
data = data.drop(['Left', 'Attrition'], axis='columns')

num = []
str = []

graph_count = 1

for i in columns:
    
    print(f'Reviewing column {i}...')
    print(list(set(data[i]))[:10])

    if type(list(set(data[i]))[1]) == type(1) or type(list(set(data[i]))[1]) == type(0.1):
        num.append(i)
        print('Type Numerical')

        sns.regplot(x=data[i], y=data['Stayed'], ci=False, line_kws={'color':'red'}) 
        plt.savefig(f'{graph_count} - Regression Plot - {i} and Stay Probability.png')

        graph_count += 1
    else:
        str.append(i)
        print('Type String')

        print('Means ', data.groupby(i)['Stayed'].mean())
        print('Counts ', data.groupby(i)['Stayed'].count())

    plt.clf()

print(num)
print(str)

# Notes from review
# Performance Rating - low performance predicts, but higher almost no effect
# Number of Promotions - good predictor
# Overtime - needs converting to binary variable - no overtime increases chance of staying
# Distance from Home - strong predictor (closer = reduced leaving chance)
# Education Level - PhD likely to stay, otherwise limited link - needs converting to binary
# Marital Status - married and divorced likely to stay - needs converting to binary
# Number of Dependents - strong link
# Job Level - higher seniority likely to stay
# Company Size - almost no link with size
# Company Tenure - older companies have slightly more loyalty
# Remote Work - remote work increases loyalty
# Leadership Opportunities - slight improvement, but quite small - needs converting to binary
# Innovation Opportunities - slight improvement, but quite small - needs converting to binary
# Company Reputation - some improvement, but quite small - needs converting to binary
# Employee Recognition - almost no relationship 
# Work-Life Balance - strong predictor

# Further Examination
# Attrition
print(data.groupby('Stayed')['Stayed'].count()) #More Stayed than left - will need to balance later on

# Age
data.Age.plot.hist(bins=5)
plt.savefig(f'{graph_count} - Histogram - Age Distribution')
plt.clf()

graph_count += 1
print(data.groupby('Age')['Age'].count()) # Distribution is Flat 
data['age_bracket'] = data['Age'].apply(lambda x: x / 10).apply(np.floor)
print(data.groupby('age_bracket')['Stayed'].mean()) # Slight increase in stay probability as people get older
data.groupby(['age_bracket'])['Stayed'].mean().plot(kind='bar')
plt.savefig(f'{graph_count} - Bar Chart - Relationship of Age with Stay Probability')
plt.clf()
graph_count += 1
data = data.drop(['age_bracket'], axis='columns')

# Gender
print(data.groupby('Gender')['Stayed'].mean()) #  Men are more likely to stay

# Years at Company
data['experience_bracket'] = data['Years at Company'].apply(lambda x: x / 10).apply(np.floor)

print(data.groupby('experience_bracket')['Stayed'].mean()) # Slight increase in stay probability as people stay longer 
data = data.drop(['experience_bracket'], axis='columns')
data['Years at Company'].plot.hist(bins=5) # Linearly related 
plt.savefig(f'{graph_count} - Histogram - Years at Company Distribution')
plt.clf()
graph_count += 1
sns.regplot(x=data['Years at Company'], y=data['Age'], ci=False, line_kws={'color':'red'}) # Also linear relationship with age
plt.savefig(f'{graph_count} - Regression Plot - Years at Company and Age.png')
plt.clf()
graph_count += 1

# Job Role
print(set(data['Job Role'])) # {'Finance', 'Media', 'Healthcare', 'Technology', 'Education'}
print(data.groupby('Job Role')['Stayed'].mean()) # Almost no impact - will take care with this feature

# Education     4498.584948
# Finance       8497.713775
# Healthcare    8001.337267
# Media         5991.803426
# Technology    9108.653511

# Monthly Income
sns.regplot(x=data['Monthly Income'], y=data['Stayed'], ci=False, line_kws={'color':'red'}) # Slight increase in stay probability with higher income, but only small effect
plt.savefig(f'{graph_count} - Regression Plot - Monthly Income and Stay Probability.png')
plt.clf()
graph_count += 1

# Job Satisfaction
print(set(data['Job Satisfaction'])) # {'Medium', 'High', 'Low', 'Very High'}
print(data.groupby('Job Satisfaction')['Stayed'].mean()) # Job Satisfaction is a very strong feature

# Performance Rating
print(data['Performance Rating'])
print(data.groupby('Performance Rating')['Stayed'].mean()) # Some correlation 

# Number of Promotions
sns.regplot(x=data['Number of Promotions'], y=data['Stayed'], ci=False, line_kws={'color':'red'}) # Some predictive ability
plt.savefig(f'{graph_count} - Regression Plot - Number of Promotions and Stay Probability.png')
plt.clf()
graph_count += 1

data['id_bracket'] = data['Employee ID'].apply(lambda x: x / 2).apply(np.floor)
print(data.groupby('id_bracket')['Stayed'].mean()) # No link with ID number 
data.groupby(['id_bracket'])['Stayed'].mean().plot(kind='bar')
plt.savefig(f'{graph_count} - Regression Plot - Employee ID and Stay Probability.png')
plt.clf()

corr_matrix = data[['Age', 'Years at Company', 'Monthly Income', 'Number of Promotions', 'Distance from Home', 'Number of Dependents', 'Company Tenure']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='seismic')
plt.savefig('Correlation Matrix')