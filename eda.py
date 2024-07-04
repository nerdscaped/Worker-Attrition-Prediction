# Install Packages and Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('train.csv')
print(data.columns) 

print(data.isnull().sum()) # No Nulls present :)

# Setting Up Data for EDA
columns = ['Employee ID', 'Age', 'Gender', 'Years at Company', 'Job Role',
       'Monthly Income', 'Work-Life Balance', 'Job Satisfaction',
       'Performance Rating', 'Number of Promotions', 'Overtime',
       'Distance from Home', 'Education Level', 'Marital Status',
       'Number of Dependents', 'Job Level', 'Company Size', 'Company Tenure',
       'Remote Work', 'Leadership Opportunities', 'Innovation Opportunities',
       'Company Reputation', 'Employee Recognition', 'Stayed']

attrition_ind = pd.get_dummies(data['Attrition'])
data = pd.concat([data, attrition_ind], axis=1)
data = data.drop(['Left', 'Attrition'], axis='columns')

# Violin Plots
graph_count = 1

sns.violinplot(data.drop('Employee ID', axis=1))
plt.savefig(f'{graph_count} - Violin Plot')
plt.clf()
graph_count += 1

sns.violinplot(data.drop(['Employee ID', 'Monthly Income'], axis=1))
plt.savefig(f'{graph_count} - Violin Plot (Excluding Monthly Income)')
plt.clf()
graph_count += 1

num = []
str = []

# Histograms
for i in columns:
    if type(list(set(data[i]))[1]) == type(1) or type(list(set(data[i]))[1]) == type(0.1):
        try:
            sns.histplot(data[i],kde=True,stat='density',bins=30)
            plt.savefig(f'{graph_count} - Histogram - {i} Distribution')
            plt.clf()
            graph_count += 1
        except:
            continue

# Regression and Bar Plots
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

        data.groupby(i)['Stayed'].mean().plot(kind='bar')
        plt.savefig(f'{graph_count} - Bar Chart - {i} and Stay Probability.png')

        graph_count += 1

        data.groupby(i)['Stayed'].count().plot(kind='bar')
        plt.savefig(f'{graph_count} - Bar Chart - {i} and Stay Counts.png')

        graph_count += 1
        # print('Means ', data.groupby(i)['Stayed'].mean())
        # print('Counts ', data.groupby(i)['Stayed'].count())

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
#More Stayed than left - will need to balance later on
# Significant impact of job role on income
# Slight increase in stay probability with higher income, but only small effect
# Job Satisfaction is a very strong feature
# No link with ID number 

# Further Examination
# Age
data['age_bracket'] = data['Age'].apply(lambda x: x / 10).apply(np.floor)
print(data.groupby('age_bracket')['Stayed'].mean()) # Slight increase in stay probability as people get older
data.groupby(['age_bracket'])['Stayed'].mean().plot(kind='bar')
plt.savefig(f'{graph_count} - Bar Chart - Relationship of Age with Stay Probability')
plt.clf()
graph_count += 1

# Years at Company
data['experience_bracket'] = data['Years at Company'].apply(lambda x: x / 10).apply(np.floor)

data.groupby(['experience_bracket'])['Stayed'].mean().plot(kind='bar')
plt.savefig(f'{graph_count} - Bar Chart - Relationship of Experience with Stay Probability')
plt.clf()
graph_count += 1

sns.regplot(x=data['Years at Company'], y=data['Age'], ci=False, line_kws={'color':'red'}) # Also linear relationship with age
plt.savefig(f'{graph_count} - Regression Plot - Years at Company and Age.png')
plt.clf()
graph_count += 1

# Job Role
print(set(data['Job Role'])) # {'Finance', 'Media', 'Healthcare', 'Technology', 'Education'}

data.groupby(['Job Role'])['Monthly Income'].mean().plot(kind='bar')
plt.savefig(f'{graph_count} - Bar Chart - Job Role and Monthly Income')
plt.clf()
graph_count += 1

# Education     4498.584948
# Finance       8497.713775
# Healthcare    8001.337267
# Media         5991.803426
# Technology    9108.653511

# Employee ID
data['id_bracket'] = data['Employee ID'].apply(lambda x: x / 2).apply(np.floor)
print(data.groupby('id_bracket')['Stayed'].mean()) # No link with ID number 
data.groupby(['id_bracket'])['Stayed'].mean().plot(kind='bar')
plt.savefig(f'{graph_count} - Regression Plot - Employee ID and Stay Probability.png')
plt.clf()

corr_matrix = data[['Age', 'Years at Company', 'Monthly Income', 'Number of Promotions', 'Distance from Home', 'Number of Dependents', 'Company Tenure']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='seismic')
plt.savefig('Correlation Matrix')

