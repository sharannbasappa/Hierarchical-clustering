import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
################################QUESTION.NO:-1###################################
Univ1 = pd.read_excel("C:\\Users\\shilpa\\Desktop\\Dataset_Assignment Clustering\\EastWestAirlines.xlsx")  ###load input file

Univ1.describe()   ###describes the given data
Univ1.info()      ##provides information about null, type of data

Univ = Univ1.drop(["ID#"], axis=1)  ##remove ID column
univ
Univ1.isna().sum()  ###check and count number of na
Univ.isnull().sum() ###check and counts number of null

Dupli = Univ1.duplicated()  ###check for duplicates
sum(dupli)                  ## sum all duplicates

uniq = Univ.drop_duplicates() ##deletes duplicates
# Normalization function 
def norm_func(i):          ###user defined fuction for normilization
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df = norm_func(Univ.iloc[:, :])  ## provided orgument, normalise all data
df.describe()                    ## re check data normalization


##plot box plot for all columns
df.describe()     

for column in df:  ##it graps box plots for all columns
    plt.figure()
    df.boxplot([column])
    
    ###Scatter plot
sns.pairplot(df.iloc[:,:])  ###for all row and colum it plots scatter plot

###outlier treatment
IQR = ["Balance"].quantile(0.75) - df["Balance"].quantile(0.25)   ##50% of data is IQR
L_limit_balance = df["Balance"].quantile(0.25) - (IQR * 1.5)      ##finding lower limit of outliers
H_limit_balance = df["Balance"].quantile(0.75) + (IQR * 1.5)      ##upper limit for outlier
df["Balance"] = pd.DataFrame(np.where(df["Balance"] > H_limit_balance , H_limit_balance ,  ##re assign after upper limit data as upper limit 
                                    np.where(df["Balance"] < L_limit_balance , L_limit_balance , df["Balance"])))  ##lower limit data as lower
seaborn.boxplot(df.Balance);plt.title('Boxplot');plt.show()  ##plot the boxplot for outlier free data                           

IQR = df["Bonus_miles"].quantile(0.75) - df["Bonus_miles"].quantile(0.25)
L_limit_Bonus_miles = df["Bonus_miles"].quantile(0.25) - (IQR * 1.5)
H_limit_Bonus_miles = df["Bonus_miles"].quantile(0.75) + (IQR * 1.5)
df["Bonus_miles"] = pd.DataFrame(np.where(df["Bonus_miles"] > H_limit_Bonus_miles , H_limit_Bonus_miles ,
                                    np.where(df["Bonus_miles"] < L_limit_Bonus_miles , L_limit_Bonus_miles , df["Bonus_miles"])))
seaborn.boxplot(df.Bonus_miles);plt.title('Boxplot');plt.show()

IQR = df["Bonus_trans"].quantile(0.75) - df["Bonus_trans"].quantile(0.25)
L_limit_Bonus_trans = df["Bonus_trans"].quantile(0.25) - (IQR * 1.5)
H_limit_Bonus_trans = df["Bonus_trans"].quantile(0.75) + (IQR * 1.5)
df["Bonus_trans"] = pd.DataFrame(np.where(df["Bonus_trans"] > H_limit_Bonus_trans , H_limit_Bonus_trans ,
                                    np.where(df["Bonus_trans"] < L_limit_Bonus_trans , L_limit_Bonus_trans , df["Bonus_trans"])))
seaborn.boxplot(df.Bonus_trans);plt.title('Boxplot');plt.show()

IQR = df["Flight_miles_12mo"].quantile(0.75) - df["Flight_miles_12mo"].quantile(0.25)
L_limit_Flight_miles_12mo = df["Flight_miles_12mo"].quantile(0.25) - (IQR * 1.5)
H_limit_Flight_miles_12mo = df["Flight_miles_12mo"].quantile(0.75) + (IQR * 1.5)
df["Flight_miles_12mo"] = pd.DataFrame(np.where(df["Flight_miles_12mo"] > H_limit_Flight_miles_12mo , H_limit_Flight_miles_12mo ,
                                    np.where(df["Flight_miles_12mo"] < L_limit_Flight_miles_12mo , L_limit_Flight_miles_12mo , df["Flight_miles_12mo"])))
seaborn.boxplot(df.Flight_miles_12mo);plt.title('Boxplot');plt.show()

IQR = df["Flight_trans_12"].quantile(0.75) - df["Flight_trans_12"].quantile(0.25)
L_limit_Flight_trans_12 = df["Flight_trans_12"].quantile(0.25) - (IQR * 1.5)
H_limit_Flight_trans_12 = df["Flight_trans_12"].quantile(0.75) + (IQR * 1.5)
df["Flight_trans_12"] = pd.DataFrame(np.where(df["Flight_trans_12"] > H_limit_Flight_trans_12 , H_limit_Flight_trans_12 ,
                                    np.where(df["Flight_trans_12"] < L_limit_Flight_trans_12 , L_limit_Flight_trans_12 , df["Flight_trans_12"])))
seaborn.boxplot(df.Flight_trans_12);plt.title('Boxplot');plt.show()

IQR = df["Days_since_enroll"].quantile(0.75) - df["Days_since_enroll"].quantile(0.25)
L_limit_Days_since_enroll = df["Days_since_enroll"].quantile(0.25) - (IQR * 1.5)
H_limit_Days_since_enroll = df["Days_since_enroll"].quantile(0.75) + (IQR * 1.5)
df["Days_since_enroll"] = pd.DataFrame(np.where(df["Days_since_enroll"] > H_limit_Days_since_enroll , H_limit_Days_since_enroll ,
                                    np.where(df["Days_since_enroll"] < L_limit_Days_since_enroll , L_limit_Days_since_enroll , df["Days_since_enroll"])))
seaborn.boxplot(df.Days_since_enroll);plt.title('Boxplot');plt.show()
# for creating dendrogram 
import gower   ####import package
from scipy.cluster.hierarchy import linkage ###imoprt package 
import scipy.cluster.hierarchy as sch ###import package

z_c = linkage(df_norm, method = "complete", metric = "euclidean") ##used complete method and eucidean distance 

# Dendrogram
plt.figure(figsize=(15, 8));  ### plot dendogram with 15 and 8 size
plt.title('Hierarchical Clustering Dendrogram');##title for dendogram
plt.xlabel('Index');   ##x labling
plt.ylabel('Distance')  ##y labling
sch.dendrogram(z,
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()
Z_a = linkage(df_norm , method="average" ,metric="euclidean")   ##using averange method
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram for average linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(linkage_average , leaf_rotation= 0 , leaf_font_size= 10)
plt.show()



# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm)  ##using complete method 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)  ###lable encoding

Univ['clust'] = cluster_labels # creating a new column and assigning it to new column 
Univ
Univ1 = Univ.iloc[:, [11,0,1,2,3,4,5,6,7,8,9,10]]  ###assigning cluster in forst column
Univ1
Univ1.head()

# Aggregate mean of each cluster
Univ1.iloc[:, 1:].groupby(Univ1.clust).mean()  #calculated mean of each column with different discriptions

# creating a csv file 
Univ1.to_csv("University.csv", encoding = "utf-8")  ###convert df into csv file

import os  
os.getcwd()   ##puting into liabrary 


######################################QUESTION NO:-2###################################

Univ1 = pd.read_excel("C:\\Users\\shilpa\\Desktop\\crime_data.xlsx")

Univ1.describe()
Univ1.info()

Univ1.isna().sum()
Univ.isnull().sum()

Dupli = Univ1.duplicated()
sum(dupli)

dropedd = dupli.drop_duplicates()
# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df = norm_func(Univ.iloc[:, :])
df.describe()

df.describe()  


##plot box plot for all columns


for column in df:
    plt.figure()
    df.boxplot([column])
    
##scatter plots 
    sns.pairplot(df.iloc[:,:])
    
    ###outlier Treatment

IQR = df["Rape"].quantile(0.75) - df["Rape"].quantile(0.25)
L_limit_Rape = df["Rape"].quantile(0.25) - (IQR * 1.5)
H_limit_Rape = df["Rape"].quantile(0.75) + (IQR * 1.5)
df["Rape"] = pd.DataFrame(np.where(df["Rape"] > H_limit_Rape , H_limit_Rape ,
                                    np.where(df["Rape"] < L_limit_Rape , L_limit_Rape , df["Rape"])))
seaborn.boxplot(df.Rape)
;plt.title('Boxplot');
plt.show()


crime_complete = AgglomerativeClustering(n_clusters=3 , linkage="complete" , affinity="euclidean").fit(df_norm)
cluster_crime_complete = pd.Series(crime_complete.labels_)
df["cluster"] = cluster_crime_complete

crime_single = AgglomerativeClustering(n_clusters=3 , linkage="single" , affinity="euclidean").fit(df_norm)
cluster_crime_single = pd.Series(crime_single.labels_)
df["cluster"]  = cluster_crime_single



crime = df.iloc[: , [5 , 0 , 1 , 2 , 3 , 4]]
df.iloc[: , 1:].groupby(crime_data.cluster).mean()

import os

df.to_csv("final_crime_data.csv" , encoding="utf-8")

os.getcwd()




##################################QUESTION:-3#######################################

Telicom_dataip = pd.read_excel("C:\\Users\\shilpa\\Desktop\\Dataset_Assignment Clustering\\Telco_customer_churn.xlsx")


Telicom_dataip.drop(['Count' , 'Quarter'] , axis=1 , inplace=True)

new_Telicom_dataip = pd.get_dummies(Telicom_dataip)

dupis = Telicom_dataip.duplicated()
sum(dupis)

Telicom_dataip = Telicom_dataip.drop_duplicates()

from sklearn.preprocessing import  OneHotEncoder

OH_enc = OneHotEncoder()

tnew = pd.DataFrame(OH_enc.fit_transform(Telicom_dataip).toarray())

from sklearn.preprocessing import  LabelEncoder
L_enc = LabelEncoder()
Telicom_dataip['Referred a Friend'] = L_enc.fit_transform(Telicom_dataip['Referred a Friend'])
Telicom_dataip['Offer'] = L_enc.fit_transform(Telicom_dataip['Offer'])
Telicom_dataip['Phone Service'] = L_enc.fit_transform(Telicom_dataip['Phone Service'])
Telicom_dataip['Multiple Lines'] = L_enc.fit_transform(Telicom_dataip['Multiple Lines'])
Telicom_dataip['Internet Service'] = L_enc.fit_transform(Telicom_dataip['Internet Service'])
Telicom_dataip['Internet Type'] = L_enc.fit_transform(Telicom_dataip['Internet Type'])
Telicom_dataip['Online Security'] = L_enc.fit_transform(Telicom_dataip['Online Security'])
Telicom_dataip['Online Backup'] = L_enc.fit_transform(Telicom_dataip['Online Backup'])
Telicom_dataip['Device Protection Plan'] = L_enc.fit_transform(Telicom_dataip['Device Protection Plan'])
Telicom_dataip['Premium Tech Support'] = L_enc.fit_transform(Telicom_dataip['Premium Tech Support'])
Telicom_dataip['Streaming TV'] = L_enc.fit_transform(Telicom_dataip['Streaming TV'])
Telicom_dataip['Streaming Movies'] = L_enc.fit_transform(Telicom_dataip['Streaming Movies'])
Telicom_dataip['Streaming Music'] = L_enc.fit_transform(Telicom_dataip['Streaming Music'])
Telicom_dataip['Unlimited Data'] = L_enc.fit_transform(Telicom_dataip['Unlimited Data'])
Telicom_dataip['Contract'] = L_enc.fit_transform(Telicom_dataip['Contract'])
Telicom_dataip['Paperless Billing'] = L_enc.fit_transform(Telicom_dataip['Paperless Billing'])
Telicom_dataip['Payment Method'] = L_enc.fit_transform(Telicom_dataip['Payment Method'])

Telicom_dataip.isna().sum()

Telicom_dataip.columns

seaborn.boxplot(Telicom_dataip["Tenure in Months"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(Telicom_dataip["Avg Monthly Long Distance Charges"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(Telicom_dataip["Avg Monthly GB Download"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(Telicom_dataip["Monthly Charge"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(Telicom_dataip["Total Charges"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(Telicom_dataip["Total Refunds"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(Telicom_dataip["Total Extra Data Charges"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(Telicom_dataip["Total Long Distance Charges"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(Telicom_dataip["Total Revenue"]);plt.title("Boxplot");plt.show()

plt.scatter(Telicom_dataip["Tenure in Months"] , Telicom_dataip["Total Extra Data Charges"])
plt.scatter(Telicom_dataip["Monthly Charge"] , Telicom_dataip["Avg Monthly Long Distance Charges"])
plt.scatter(Telicom_dataip["Total Long Distance Charges"] , Telicom_dataip["Total Revenue"])

IQR = Telicom_dataip["Avg Monthly GB Download"].quantile(0.75) - Telicom_dataip["Avg Monthly GB Download"].quantile(0.25)
L_limit_Avg_Monthly_GB_Download = Telicom_dataip["Avg Monthly GB Download"].quantile(0.25) - (IQR * 1.5)
H_limit_Avg_Monthly_GB_Download = Telicom_dataip["Avg Monthly GB Download"].quantile(0.75) + (IQR * 1.5)
Telicom_dataip["Avg Monthly GB Download"] = pd.DataFrame(np.where(Telicom_dataip["Avg Monthly GB Download"] > H_limit_Avg_Monthly_GB_Download , H_limit_Avg_Monthly_GB_Download ,
                                    np.where(Telicom_dataip["Avg Monthly GB Download"] < L_limit_Avg_Monthly_GB_Download , L_limit_Avg_Monthly_GB_Download , Telicom_dataip["Avg Monthly GB Download"])))
seaborn.boxplot(Telicom_dataip["Avg Monthly GB Download"]);plt.title('Boxplot');plt.show()

IQR = Telicom_dataip["Total Refunds"].quantile(0.75) - Telicom_dataip["Total Refunds"].quantile(0.25)
L_limit_Total_Refunds = Telicom_dataip["Total Refunds"].quantile(0.25) - (IQR * 1.5)
H_limit_Total_Refunds = Telicom_dataip["Total Refunds"].quantile(0.75) + (IQR * 1.5)
Telicom_dataip["Total Refunds"] = pd.DataFrame(np.where(Telicom_dataip["Total Refunds"] > H_limit_Total_Refunds , H_limit_Total_Refunds ,
                                    np.where(Telicom_dataip["Total Refunds"] < L_limit_Total_Refunds , L_limit_Total_Refunds , Telicom_dataip["Total Refunds"])))
seaborn.boxplot(Telicom_dataip["Total Refunds"]);plt.title('Boxplot');plt.show()

IQR = Telicom_dataip["Total Extra Data Charges"].quantile(0.75) - Telicom_dataip["Total Extra Data Charges"].quantile(0.25)
L_limit_Total_Extra_Data_Charges = Telicom_dataip["Total Extra Data Charges"].quantile(0.25) - (IQR * 1.5)
H_limit_Total_Extra_Data_Charges = Telicom_dataip["Total Extra Data Charges"].quantile(0.75) + (IQR * 1.5)
Telicom_dataip["Total Extra Data Charges"] = pd.DataFrame(np.where(Telicom_dataip["Total Extra Data Charges"] > H_limit_Total_Extra_Data_Charges , H_limit_Total_Extra_Data_Charges ,
                                    np.where(Telicom_dataip["Total Extra Data Charges"] < L_limit_Total_Extra_Data_Charges , L_limit_Total_Extra_Data_Charges , Telicom_dataip["Total Extra Data Charges"])))
seaborn.boxplot(Telicom_dataip["Total Extra Data Charges"]);plt.title('Boxplot');plt.show()

IQR = Telicom_dataip["Total Long Distance Charges"].quantile(0.75) - Telicom_dataip["Total Long Distance Charges"].quantile(0.25)
L_limit_Total_Long_Distance_Charges = Telicom_dataip["Total Long Distance Charges"].quantile(0.25) - (IQR * 1.5)
H_limit_Total_Long_Distance_Charges = Telicom_dataip["Total Long Distance Charges"].quantile(0.75) + (IQR * 1.5)
Telicom_dataip["Total Long Distance Charges"] = pd.DataFrame(np.where(Telicom_dataip["Total Long Distance Charges"] > H_limit_Total_Long_Distance_Charges , H_limit_Total_Long_Distance_Charges ,
                                    np.where(Telicom_dataip["Total Long Distance Charges"] < L_limit_Total_Long_Distance_Charges , L_limit_Total_Long_Distance_Charges , Telicom_dataip["Total Long Distance Charges"])))
seaborn.boxplot(Telicom_dataip["Total Long Distance Charges"]);plt.title('Boxplot');plt.show()

IQR = Telicom_dataip["Total Revenue"].quantile(0.75) - Telicom_dataip["Total Revenue"].quantile(0.25)
L_limit_Total_Revenue = Telicom_dataip["Total Revenue"].quantile(0.25) - (IQR * 1.5)
H_limit_Total_Revenue = Telicom_dataip["Total Revenue"].quantile(0.75) + (IQR * 1.5)
Telicom_dataip["Total Revenue"] = pd.DataFrame(np.where(Telicom_dataip["Total Revenue"] > H_limit_Total_Revenue , H_limit_Total_Revenue ,
                                    np.where(Telicom_dataip["Total Revenue"] < L_limit_Total_Revenue , L_limit_Total_Revenue , Telicom_dataip["Total Revenue"])))
seaborn.boxplot(Telicom_dataip["Total Revenue"]);plt.title('Boxplot');plt.show()

def std_fun(i):
    x = (i-i.mean()) / (i.std())
    return (x)

Telicom_dataip_norm = std_fun(new_Telicom_dataip)

str(Telicom_dataip_norm)

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage

telco_single_linkage = linkage(Telicom_dataip_norm , method="single" , metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using single linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(telco_single_linkage , leaf_font_size=10 ,leaf_rotation=0)
plt.show()

telco_complete_linkage = linkage(telco_single_linkage , method="complete" , metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using complete linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(telco_single_linkage , leaf_font_size=10 ,leaf_rotation=0)
plt.show()

telco_average_linkage = linkage(telco_complete_linkage , method="average" , metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using single linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(telco_average_linkage , leaf_font_size=10 ,leaf_rotation=0)
plt.show()

telco_centroid_linkage = linkage(Telicom_dataip_norm , method="centroid" , metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using single linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(telco_centroid_linkage , leaf_font_size=10 ,leaf_rotation=0)
plt.show()

from sklearn.cluster import  AgglomerativeClustering

telco_single = AgglomerativeClustering(n_clusters=3 , linkage="single" , affinity="euclidean").fit(Telicom_dataip_norm)
cluster_telco_single = pd.Series(telco_single.labels_)
Telicom_dataip["cluster"] = cluster_telco_single

telco_complete = AgglomerativeClustering(n_clusters=3 , linkage="complete" , affinity="euclidean").fit(Telicom_dataip_norm)
cluster_telco_complete = pd.Series(telco_complete.labels_)
Telicom_dataip["cluster"] = cluster_telco_complete

telco_average = AgglomerativeClustering(n_clusters=3 , linkage="average" , affinity="euclidean").fit(Telicom_dataip_norm)
cluster_telco_average = pd.Series(telco_average.labels_)
Telicom_dataip["cluster"] = cluster_telco_average

telco_centroid = AgglomerativeClustering(n_clusters=3 , linkage="centroid" ,  affinity="euclidean").fit(Telicom_dataip_norm)
cluster_telco_centroid = pd.Series(telco_centroid.labels_)
Telicom_dataip["cluster"] = cluster_telco_centroid

Telicom_dataip.iloc[: , 0:29].groupby(Telicom_dataip.cluster).mean()

import os

Telicom_dataip.to_csv("final_Telicom_dataip.csv" , encoding="utf-8")

os.getcwd()

import  gower
from scipy.cluster.hierarchy import fcluster , dendrogram
gowers_matrix = gower.gower_matrix(Telicom_dataip)
gowers_linkage = linkage(gowers_matrix)
gcluster = fcluster(gowers_linkage , 3 , criterion = 'maxclust')
dendrogram(gowers_linkage)
Telicom_dataip["cluster"] = gcluster
Telicom_dataip.iloc[: , 0:29].groupby(Telicom_dataip.cluster).mean()

import os

Telicom_dataip.to_csv("final2_Telicom_dataip.csv" , encoding="utf-8")

os.getcwd()


#####################################Question_4##################################
audata = pd.read_excel("C:\\Users\\shilpa\\Desktop\\Dataset_Assignment Clustering\\AutoInsurance.csv")

audata.drop(['Customer'] , axis= 1 , inplace = True)

new_audata = audata.iloc[ : ,1:]

new_audata.isna().sum()

new_audata.columns

duplis = new_audata.duplicated()
sum(duplis)

new_audata = new_audata.drop_duplicates()

seaborn.boxplot(new_audata["Customer Lifetime Value"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(new_audata["Income"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(new_audata["Monthly Premium Auto"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(new_audata["Months Since Last Claim"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(new_audata["Months Since Policy Inception"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(new_audata["Total Claim Amount"]);plt.title("Boxplot");plt.show()

plt.scatter(new_audata["Customer Lifetime Value"] , new_audata["Income"])
plt.scatter(new_audata["Monthly Premium Autos"] , new_audata["Months Since Last Claime"])
plt.scatter(new_audata["Months Since Policy Inception"] , new_audata["Total Claim Amount"])

IQR = new_audata["Customer Lifetime Value"].quantile(0.75) - new_audata["Customer Lifetime Value"].quantile(0.25)
L_limit_Customer_Lifetime_Value = new_audata["Customer Lifetime Value"].quantile(0.25) - (IQR * 1.5)
H_limit_Customer_Lifetime_Value = new_audata["Customer Lifetime Value"].quantile(0.75) + (IQR * 1.5)
new_audata["Customer Lifetime Value"] = pd.DataFrame(np.where(new_audata["Customer Lifetime Value"] > H_limit_Customer_Lifetime_Value , H_limit_Customer_Lifetime_Value ,
                                    np.where(new_audata["Customer Lifetime Value"] < L_limit_Customer_Lifetime_Value , L_limit_Customer_Lifetime_Value , new_audata["Customer Lifetime Value"])))
seaborn.boxplot(new_audata["Customer Lifetime Value"]);plt.title('Boxplot');plt.show()

IQR = new_audata["Monthly Premium Auto"].quantile(0.75) - new_audata["Monthly Premium Auto"].quantile(0.25)
L_limit_Monthly_Premium_Auto = new_audata["Monthly Premium Auto"].quantile(0.25) - (IQR * 1.5)
H_limit_Monthly_Premium_Auto = new_audata["Monthly Premium Auto"].quantile(0.75) + (IQR * 1.5)
new_audata["Monthly Premium Auto"] = pd.DataFrame(np.where(new_audata["Monthly Premium Auto"] > H_limit_Monthly_Premium_Auto , H_limit_Monthly_Premium_Auto ,
                                    np.where(new_audata["Monthly Premium Auto"] < L_limit_Monthly_Premium_Auto , L_limit_Monthly_Premium_Auto , new_audata["Monthly Premium Auto"])))
seaborn.boxplot(new_audata["Monthly Premium Auto"]);plt.title('Boxplot');plt.show()

IQR = new_audata["Total Claim Amount"].quantile(0.75) - new_audata["Total Claim Amount"].quantile(0.25)
L_limit_Total_Claim_Amount = new_audata["Total Claim Amount"].quantile(0.25) - (IQR * 1.5)
H_limit_Total_Claim_Amount = new_audata["Total Claim Amount"].quantile(0.75) + (IQR * 1.5)
new_audata["Total Claim Amount"] = pd.DataFrame(np.where(new_audata["Total Claim Amount"] > H_limit_Total_Claim_Amount , H_limit_Total_Claim_Amount ,
                                    np.where(new_audata["Total Claim Amount"] < L_limit_Total_Claim_Amount , L_limit_Total_Claim_Amount , new_audata["Total Claim Amount"])))
seaborn.boxplot(new_audata["Total Claim Amount"]);plt.title('Boxplot');plt.show()

dummy_audata = pd.get_dummies(new_audata)

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

audata_norm = norm_func(dummy_audata)

from sklearn.cluster import AgglomerativeClustering

auto_single = AgglomerativeClustering(n_clusters=3 , linkage="single" , affinity="euclidean").fit(audata_norm)
cluster_auto_single = pd.Series(auto_single.labels_)
new_audata["cluster"] = cluster_auto_single

auto_complete = AgglomerativeClustering(n_clusters=3 , linkage="complete" , affinity="euclidean").fit(audata_norm)
cluster_auto_complete = pd.Series(auto_complete.labels_)
new_audata["cluster"] = cluster_auto_complete

auto_average = AgglomerativeClustering(n_clusters=3 , linkage="average" , affinity="euclidean").fit(audata_norm)
cluster_auto_average = pd.Series(auto_average.labels_)
new_audata["cluster"] = cluster_auto_average

auto_centroid = AgglomerativeClustering(n_clusters=3 , linkage="centroid" , affinity="euclidean").fit(audata_norm)
cluster_auto_centroid = pd.Series(auto_centroid.labels_)
new_audata["cluster"] = cluster_auto_centroid

new_audata.iloc[: ,:23].groupby(new_audata.cluster).mean()

import os

new_audata.to_csv("final_audata.csv" , encoding="utf-8")

os.getcwd()


