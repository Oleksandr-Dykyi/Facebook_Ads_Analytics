#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


fad = pd.read_csv("facebook_ads_data.csv")


# In[6]:


fad_2021 = fad[fad['ad_date'].str.contains('2021', case=False, na=False)]
fad_group_date = fad_2021.groupby('ad_date')
fad_date_indicators = fad_group_date['total_spend', 'total_value'].sum()


# In[7]:


plt.figure(figsize=(12, 6))
plt.plot(fad_date_indicators.index, fad_date_indicators['total_spend'])
plt.title('Daily ad spend in 2021')
plt.ylabel('Total spend')
xticks_indices = np.arange(0, len(fad_date_indicators), 8)
xticks_labels = fad_date_indicators.index[xticks_indices]
plt.xticks(xticks_indices, xticks_labels, rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[8]:


fad_date_indicators['romi'] = (fad_date_indicators['total_value'] - fad_date_indicators['total_spend']) / fad_date_indicators['total_spend']
fad_date_indicators.dropna(inplace=True)


# In[10]:


plt.figure(figsize=(12, 6))
plt.plot(fad_date_indicators.index, fad_date_indicators['romi'])
plt.title('Daily ROMI in 2021')
plt.ylabel('ROMI')
xticks_indices = np.arange(0, len(fad_date_indicators), 9)
xticks_labels = fad_date_indicators.index[xticks_indices]
plt.xticks(xticks_indices, xticks_labels, rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[11]:


window_size = 30
fad_date_indicators['mean_spend'] = fad_date_indicators['total_spend'].rolling(window=window_size).mean()
fad_date_indicators['mean_romi'] = fad_date_indicators['romi'].rolling(window=window_size).mean()


# In[12]:


plt.figure(figsize=(12, 6))
plt.plot(fad_date_indicators.index, fad_date_indicators['total_spend'], label='Spend')
plt.plot(fad_date_indicators.index, fad_date_indicators['mean_spend'], label=f'Moving average sped ({window_size}-day)', linestyle='--')
plt.title('Daily spend and moving average spend in 2021')
plt.ylabel('Spend')
xticks_indices = np.arange(0, len(fad_date_indicators), 9)
xticks_labels = fad_date_indicators.index[xticks_indices]
plt.xticks(xticks_indices, xticks_labels, rotation=90)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(fad_date_indicators.index, fad_date_indicators['romi'], label='ROMI')
plt.plot(fad_date_indicators.index, fad_date_indicators['mean_romi'], label=f'Moving average ROMI ({window_size}-day)', linestyle='--')
plt.title('Daily ROMI and Moving Average ROMI in 2021')
plt.ylabel('ROMI')
xticks_indices = np.arange(0, len(fad_date_indicators), 9)
xticks_labels = fad_date_indicators.index[xticks_indices]
plt.xticks(xticks_indices, xticks_labels, rotation=90)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[13]:


fad_group_campaign = fad.groupby('campaign_name')
fad_campaign_indicators = fad_group_campaign['total_spend', 'total_value'].sum()


# In[14]:


plt.figure(figsize=(12, 6))
plt.bar(fad_campaign_indicators.index, fad_campaign_indicators['total_spend'])
plt.title('Total ad spend in each campaign')
plt.ylabel('Total spend')
plt.tight_layout()
plt.show()


# In[15]:


fad_campaign_indicators['romi'] = (fad_campaign_indicators['total_value'] - fad_campaign_indicators['total_spend']) / fad_campaign_indicators['total_spend']


# In[16]:


plt.figure(figsize=(12, 6))
plt.bar(fad_campaign_indicators.index, fad_campaign_indicators['romi'])
plt.title('ROMI in each campaign')
plt.ylabel('ROMI')
plt.tight_layout()
plt.show()


# In[17]:


fad['Romi']=fad['romi']-1


# In[18]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='campaign_name', y='Romi', data=fad)
plt.title('Spread of daily ROMI in each campaign')
plt.xlabel('')
plt.ylabel('ROMI')
plt.tight_layout()
plt.show()


# In[19]:


plt.figure(figsize=(12, 6))
plt.hist(fad['Romi'], bins=21, edgecolor='black')
plt.title('Distribution of ROMI values')
plt.xlabel('ROMI')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# In[20]:


correlation_matrix = fad.iloc[:,:10].corr()


# In[22]:


plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation heat map')
plt.show()


# In[23]:


correlation_values = correlation_matrix.unstack().sort_values(ascending=False)
max_corr = correlation_values[correlation_values < 1].max()
min_corr = correlation_values.min()
max_corr_indices = correlation_values[correlation_values == max_corr].index
min_corr_indices = correlation_values[correlation_values == min_corr].index
max_corr_indices[0], min_corr_indices[0]


# In[24]:


total_value_corr = correlation_matrix['total_value'].drop('total_value')
highest_corr_feature = total_value_corr.idxmax()
highest_corr_feature


# In[25]:


sns.lmplot(x='total_spend', y='total_value', data=fad)
plt.title('Scatter plot with linear regression')
plt.xlabel('Total spend')
plt.ylabel('Total value')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




