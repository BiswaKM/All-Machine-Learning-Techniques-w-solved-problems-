import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
import numpy as np

retail = []
with open("C:\\Users\\biswa\\Desktop\\360DigiTMG\\15 . Module 15 - Data Mining Unsupervised Learning - Association Rules\\Assignment\\transactions_retail1.csv") as f:
    retail = f.read()

# splitting the data into separate transactions using separator as "\n"
retail = retail.split("\n")
retail_list = []
for i in retail:
    retail_list.append(i.split(","))

all_retail_list = [i for item in retail_list for i in item]

all_retail_list1=[]
for i in all_retail_list:
    if (i!='NA'):
        all_retail_list1.append(i)

from collections import Counter,OrderedDict

item_frequencies = Counter(all_retail_list1)
# after sorting
item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 

import matplotlib.pyplot as plt
%matplotlib inline
plt.bar(height = frequencies[0:11],x = list(range(0,11)),color='rgbkymc')
plt.xticks(list(range(0,11),),items[0:11],rotation=90)
plt.xlabel("items")
plt.ylabel("Count");plt.xlabel("Items")
plt.show()

# Creating Data Frame for the transactions data 

retail_series  = pd.DataFrame(pd.Series(all_retail_list1))

retail_series=pd.DataFrame(pd.Series(retail_list))
retail_series.columns = ["transactions"]
# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = retail_series['transactions'].str.join(sep=',').str.get_dummies(sep=',')

frequent_itemsets = apriori(X, min_support=0.005, max_len=3,use_colnames = True)
 
# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(0,11)),height = frequent_itemsets.support[0:11],color='rgmyk')
plt.xticks(list(range(0,11)),frequent_itemsets.itemsets[0:11],rotation=90)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending = False).head(10)
