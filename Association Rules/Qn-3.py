import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt
%matplotlib inline
phone=pd.read_csv("C:\\Users\\biswa\\Desktop\\360DigiTMG\\15 . Module 15 - Data Mining Unsupervised Learning - Association Rules\\Assignment\\myphonedata.csv") 
phone=phone.drop(columns=['V1','V2','V3'])
frequent_phone = apriori(phone, min_support=0.005, max_len=3,use_colnames = True)

# Most Frequent item sets based on support 
frequent_phone.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(0,11)),height = frequent_phone.support[0:11],color='rgmyk')
plt.xticks(list(range(0,11)),frequent_phone.itemsets[0:11],rotation=90)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_phone, metric="lift", min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending = False).head(10)

############################################### Extra part ###################################
def to_list(i):
    return (sorted(list(i)))


ma_movie = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


ma_movie = ma_movie.apply(sorted)

rules_sets = list(ma_movie)
 
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(3)


