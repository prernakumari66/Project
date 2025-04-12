import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loading dataset
df = pd.read_csv(r"C:\LPU\LPU S4\INT375\MA2020_MANIPUR.csv", encoding='ISO-8859-1')

#checking the correct loading of dataset
print(df.head())

#--------------------------------------------------------------------------------------------------------------------------------

#Exploring the dataset
pd.set_option('display.max_info_columns', 183)
print("\nInformation about the dataset:\n", df.info())
print("Statistical description of the dataset:\n", df.describe())

#--------------------------------------------------------------------------------------------------------------------------------

#Checking missing values in the dataset
missing_values = df.isnull().sum().sort_values(ascending=False)
print("\nMissing Values per Column:")
print(missing_values[missing_values > 0])

#Handling missing values (cleaning the dataset)

## droping columns with missing values 50% or more
df = df.drop(['AVAILABILITY OF DRAINAGE FACILITIES','DOES THE VILLAGE HAS ANY FARMERS COLLECTIVE','NUMBER OF HOUSEHOLDS ENGAGED IN COTTAGE AND SMALL SCALE UNITS','NUMBER OF HOUSEHOLDS ELECTRIFIED BY SOLAR ENERGY/WIND ENERGY','OTHER ASSEMBLY CONSTITUENCIES','NUMBER OF HOUSEHOLDS WHERE ONLY SOURCE OF LIVELIHOOD IS MINOR FOREST PRODUCTION','AVAILABILITY OF RECREATIONAL CENTRE/SPORTS PLAYGROUND ETC','NUMBER OF HOUSEHOLDS REGISTERED FOR HEALTH INSURANCE SERVICES UNDER PRADHAN MANTRI JAN AROGYA YOJANA (PMJAY)/STATE SPECIFIC HEALTH INSURANCE SCHEMES'],axis=1)

## filling categorical columns with missing values with mode
df['AVAILABILITY OF ELECTRICITY FOR DOMESTIC USE']= df['AVAILABILITY OF ELECTRICITY FOR DOMESTIC USE'].fillna(df['AVAILABILITY OF ELECTRICITY FOR DOMESTIC USE'].mode()[0])
df['AVAILABILITY OF TELEPHONE SERVICES']= df['AVAILABILITY OF TELEPHONE SERVICES'].fillna(df['AVAILABILITY OF TELEPHONE SERVICES'].mode()[0])

## Removing rows with missing values in 'AC CODE' column
df = df.dropna(subset=['AC CODE'])

# remove duplicates rows if present
df = df.drop_duplicates()

# Checking that all missing values is handled or not
print("\nAfter handling missing values, Total missing values in the dataset: ",df.isnull().sum().sum())

#--------------------------------------------------------------------------------------------------------------------------------

#performing basic operations
print("\nBasic Operations:\n")
print("\nHead of the dataset:\n", df.head(5))
print("\nTail of the dataset:\n", df.tail(5))
print("\nShape of the dataset: ", df.shape)
print("Datatypes of dataset:\n", df.dtypes)
print("Actual column names :\n")
print(df.columns.tolist())
print("Number of unique values in each column:\n", df.nunique())

df.to_csv("Manipur2020_cleaned.csv", index=False)
#--------------------------------------------------------------------------------------------------------------------------------

# Objective 1 : Demographic Analysis

# Gender Ratio
df['Gender Ratio'] = (df['NUMBER OF FEMALE'] / df['NUMBER OF MALE']) * 1000
# Plot Gender Ratio Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Gender Ratio'], kde=True, bins=20, color='purple')
plt.title('Distribution of Gender Ratio (Females per 1000 Males)')
plt.xlabel('Gender Ratio')
plt.ylabel('Number of Villages')
plt.grid(True)
plt.show()

# Literacy Ratio
df['Literacy Ratio'] = (df['NUMBER OF GRADUATES/POST GRADUATES IN THE VILLAGE'] / df['NUMBER OF TOTAL POPULATION']) * 100
# Plot Literacy Ratio Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Literacy Ratio'].dropna(), kde=True, bins=20, color='teal')
plt.title('Distribution of Literacy Ratio (Graduates per 100 people)')
plt.xlabel('Literacy Ratio (%)')
plt.ylabel('Number of Villages')
plt.grid(True)
plt.show()

#Box plot for Literacy Ratio by District
plt.figure(figsize=(14, 6))
sns.boxplot(x='DISTRICT NAME', y='Literacy Ratio', data=df,showfliers=False,patch_artist=True)
plt.xticks(rotation=45)
plt.title('Literacy Ratio Distribution by District')
plt.tight_layout()
plt.show()

#Scatter Plot — Gender Ratio vs Literacy Ratio
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Gender Ratio', y='Literacy Ratio', hue='DISTRICT NAME', data=df)
plt.title('Gender Ratio vs Literacy Ratio')
plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------

# Objective 2 : Livelihood Analysis


livelihood_df = df[['NUMBER OF HOUSEHOLDS ENGAGED MAJORLY IN FARM ACTIVITIES','NUMBER OF HOUSEHOLDS ENGAGED MAJORLY IN NON-FARM ACTIVITIES']].sum()

plt.figure(figsize=(7, 7))
colors = ['mediumseagreen', 'sandybrown']
plt.pie(livelihood_df,labels=['Farm', 'Non-Farm'],autopct='%1.1f%%',startangle=140,colors=colors,wedgeprops=dict(width=0.4))
plt.title('Livelihood Sources')
plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------

# Objective 3 : Plotting graph on Top 10 Districts by number of Farmers

districtwise_stats = df.groupby("DISTRICT NAME")[["TOTAL NUMBER OF FARMERS ","NUMBER OF FARMERS RECEIVED BENEFITS UNDER PMFBY  (PRADHAN MANTRI FASAL BIMA YOJANA )","NUMBER OF FARMERS ADOPTED ORGANIC FARMING DURING 2018-19"]].sum().sort_values(by="TOTAL NUMBER OF FARMERS ", ascending=False).head(10)

districtwise_stats = districtwise_stats.reset_index()
fig, axs = plt.subplots(1, 3, figsize=(22, 6))

# Total Farmers
sns.barplot(data=districtwise_stats,x="DISTRICT NAME",y="TOTAL NUMBER OF FARMERS ",hue="DISTRICT NAME",palette="Blues_d",legend=False,ax=axs[0])
axs[0].set_title("Top 10 Districts by Total Farmers")
axs[0].tick_params(axis='x', rotation=45)
axs[0].set_xlabel("")
axs[0].set_ylabel("Number of Farmers")

# Farmers under PMFBY
sns.barplot(data=districtwise_stats,x="DISTRICT NAME",y="NUMBER OF FARMERS RECEIVED BENEFITS UNDER PMFBY  (PRADHAN MANTRI FASAL BIMA YOJANA )",hue="DISTRICT NAME",palette="Oranges_d",legend=False,ax=axs[1])
axs[1].set_title("PMFBY Beneficiaries by District")
axs[1].tick_params(axis='x', rotation=45)
axs[1].set_xlabel("")
axs[1].set_ylabel("PMFBY Beneficiaries")

# Organic Farmers
sns.barplot(data=districtwise_stats, x="DISTRICT NAME",y="NUMBER OF FARMERS ADOPTED ORGANIC FARMING DURING 2018-19",hue="DISTRICT NAME",palette="Greens_d",legend=False,ax=axs[2])
axs[2].set_title("Organic Farming Adoption by District")
axs[2].tick_params(axis='x', rotation=45)
axs[2].set_xlabel("")
axs[2].set_ylabel("Number of Organic Farmers")

plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------

#Objective 4 : Population vs. Households (Scatter Plot)

plt.figure(figsize=(8, 5))
sns.scatterplot(x='NUMBER OF TOTAL HOUSEHOLD',y='NUMBER OF TOTAL POPULATION',data=df,alpha=0.6,edgecolor='w')
plt.title('Population vs. Number of Households')
plt.xlabel('Total Households')
plt.ylabel('Total Population')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------

# Objective 5 : Availability of Financial Services (Donut Chart)

service_cols = ['AVAILABILITY OF BANKS','AVAILABILITY OF ATM','AVAILABILITY OF POST OFFICE/SUB-POST OFFICE']


service_data = [(df[col].str.strip().str.lower() == 'yes').mean() * 100 for col in service_cols]
labels = ['Banks', 'ATM', 'Post Office']

plt.figure(figsize=(7, 7))
# plotting donut chart
wedges, texts, autotexts = plt.pie(service_data,labels=labels,autopct='%1.1f%%',startangle=140,wedgeprops=dict(width=0.4),colors=sns.color_palette('pastel'))
plt.setp(autotexts, size=12, weight="bold")
plt.title('Availability of Financial Services')
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------

# Objective 6: Training of Elected Representatives (Bar Plot)

training_data = df.groupby('BLOCK NAME')[['NUMBER OF ELECTED REPRESENTATIVES ORIENTED UNDER RASHTRIYA GRAM SWARAJ ABHIYAN',
                                          'NUMBER OF ELECTED REPRESENTATIVES UNDERGONE REFRESHER TRAINING UNDER RASHTRIYA GRAM SWARAJ ABHIYAN']].sum().sort_values(by='NUMBER OF ELECTED REPRESENTATIVES ORIENTED UNDER RASHTRIYA GRAM SWARAJ ABHIYAN', ascending=False).head(10)
training_data.plot(kind='bar', figsize=(12,6))
plt.title("Top 10 Blocks: Orientation & Training of Elected Representatives")
plt.ylabel("Number of Representatives")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------

# Objective 7: Correlation Between Budget and Infrastructure (Heatmap)

cols = ['TOTAL APPROVED LABOUR BUDGET FOR THE YEAR 2018-19','TOTAL AREA COVERED UNDER IRRIGATION (DRIP, SPRINKLER), IF IN ACRES DIVIDE BY 2.47',
    'NUMBER OF HOUSEHOLDS HAVING PIPED WATER CONNECTION ','NUMBER OF FARMERS RECEIVED THE SOIL TESTING REPORT']

corr_matrix = df[cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix: Budget vs Infrastructure", fontsize=14)
plt.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------


