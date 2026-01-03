#!/usr/bin/env python
# coding: utf-8

# In[133]:


import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns


# In[158]:


df=pd.read_csv("D:/DATA SCIENCE PROJECT/LR/insurance.csv") # raed csv
df


# In[135]:


df.head(5) # check top five data


# In[136]:


df.columns # check columns
df.shape # 


# In[137]:


df.info() #summary of dataset 


# In[138]:


df.describe().round() # statistical summary


# In[125]:


#checking distribution of age
sns.distplot(x=df["age"])

sns.show()


# In[139]:


# feature selection
# direct use corr with heatmap
sns.heatmap(df.corr(), Annot=True)
plt.show()


# In[159]:


# as we are gettinng error could not convert string to float: 'male' instaed of drop we will convert male=1 and female=0
# same as smoker and region is not imp so dropping that column

df["sex"]=df["sex"].replace({"male":1,"female":0})
df["smoker"]=df["smoker"].replace({"yes":1,"no":0})
df=df.drop("region",axis=1)

df


# In[141]:


# direct use corr with heatmap

sns.heatmap(df.corr(),annot=True,annot_kws={"size": 7})


# In[160]:


# above graph shows wean corr between charges with sex and childern so we dropping that col
df.drop(columns=['sex', 'children',"age","bmi"], inplace=True)


# In[161]:


df


# In[162]:


#check null
df.isnull().values.any()
# how many
df.isnull().sum()


# In[163]:


# as not so much null vlues we can drop that values
df=df.dropna()
# we are no considering outliers but
#if we want to replace null values for normal dist repalce with mean and skewed dist use median 
#df["age"]=df["age"].fillna(df["age"].mean())
#df["bmi"]=df["bmi"].fillna(df["bmi"].mean())


# In[164]:


df


# In[148]:


df.isnull().sum()


# In[87]:


# now EDA 
#Check if charges is normally distributed
sns.histplot(df["charges"])
#right skewed


# In[89]:


# we will check any relation between smoker and charges
sns.barplot(x=df["smoker"], y=df["charges"])
#means smoker has high charges


# In[90]:


# we will check any realtion between age and charges
sns.scatterplot(x=df["age"],y=df["charges"])
#means charges increases as age increases


# In[92]:


#check past consultation and charges
sns.scatterplot(x=df["past_consultations"],y=df["charges"])
# linear


# In[ ]:


# any link between past hospilization and charges


# In[94]:


sns.barplot(x=df["NUmber_of_past_hospitalizations"],y=df["charges"])
# again linear


# In[95]:


# check outliers for charges
sns.boxplot(df["charges"])
# yes they have outliers in charges, annual salary , hospital_exp, past cons , but wr are treating as outliers as they have impact from independent varible


# In[96]:


# we are no considering outliers but
#if we want to replace null values for normal dist repalce with mean and skewed dist use median 
#df["age"]=df["age"].fillna(df["age"].mean())
#df["bmi"]=df["bmi"].fillna(df["bmi"].mean())




# In[149]:


#Feature selection used by corr
df.corr()


# In[202]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Step 1: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Scale features
sc_X = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)  # only transform, don't fit again

# Step 3: Scale target (optional, recommended for neural nets; for linear regression, you can skip)
sc_y = StandardScaler()
y_train_scaled = sc_y.fit_transform(y_train.to_numpy().reshape(-1, 1))  # convert to 2D

# Step 4: Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)  # use scaled y

# Step 5: Predict on test set
y_pred_scaled = model.predict(X_test_scaled)  # 1D array or 2D depending on model

# Step 6: Inverse-transform predictions to original scale
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Step 7: Now y_pred is in the original target scale
print(y_pred)


# In[208]:


from sklearn.metrics import *
r2=r2_score(y_test,y_pred)
print(r2)


# In[220]:


comp=pd.DataFrame({"actual":y_test,"predicted":y_pred})
comp["error"]=comp["actual"]-comp["predicted"]
comp


# In[222]:


sns.scatterplot(x=comp["actual"],y=comp["predicted"])


# In[231]:


#goodness of fit
sns.scatterplot(x=comp ["actual"], y=comp ["predicted"])

lims = [
    np.min([comp ["actual"].min(), comp ["predicted"].min()]),
    np.max([comp ["actual"].max(), comp ["predicted"].max()])
]
plt.plot(lims, lims, 'r--') 
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("actual vs predicted")
plt.show()


# In[ ]:




