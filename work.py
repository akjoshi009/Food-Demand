import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt



train=pd.read_csv(r"D:\24Projects\Food demand Forcast\train.csv")
fulfilment_center_info=pd.read_csv(r"D:\24Projects\Food demand Forcast\fulfilment_center_info.csv")
meal_info=pd.read_csv(r"D:\24Projects\Food demand Forcast\meal_info.csv")
test=pd.read_csv(r"D:\24Projects\Food demand Forcast\test_QoiMO9B.csv")




train=train.merge(fulfilment_center_info,on="center_id",how='left')
train=train.merge(meal_info,on="meal_id",how='left')


test=test.merge(fulfilment_center_info,on="center_id",how='left')
test=test.merge(meal_info,on="meal_id",how='left')

train.columns

le = preprocessing.LabelEncoder()
le.fit(train["center_type"])
train["center_typech"]=le.fit_transform(train["center_type"])
test["center_typech"]=le.fit_transform(test["center_type"])

le = preprocessing.LabelEncoder()
le.fit(train["category"])
train["categorych"]=le.fit_transform(train["category"])
test["categorych"]=le.fit_transform(test["category"])

le = preprocessing.LabelEncoder()
le.fit(train["cuisine"])
train["cuisinech"]=le.fit_transform(train["cuisine"])
test["cuisinech"]=le.fit_transform(test["cuisine"])


train["num_orders2"]=np.log(train["num_orders"])
train.columns

def changetomonth(txt):
    if (txt>52):
        if(txt-52>52):
            return changetomonth(txt-52)
        else:
            return int(txt-52)
    else:
        return txt

#train["Updated_week"].value_counts()

train["Updated_week"]=train["week"].apply(changetomonth)
test["Updated_week"]=test["week"].apply(changetomonth)






train["Month"]=round(train["Updated_week"]/4)
test["Month"]=round(test["Updated_week"]/4)

train["Quarter"]=round(train["Month"]/3)
test["Quarter"]=round(test["Month"]/3)



train["year"]=round(train["week"]/52)
test["year"]=round(test["week"]/52)

train["twoquarters"]=round(train["week"]/26)
test["twoquarters"]=round(test["week"]/26)

train["twoquarters"]=round(train["week"]/26)
test["twoquarters"]=round(test["week"]/26)

train["tquarters"]=round(train["week"]/10)
test["tquarters"]=round(test["week"]/10)



def pricehighlow(txt,mn):
    if(txt>mn):
        return 1
    else:
        return 2
  
    
bsmean=train['base_price'].median()
chmean=train['checkout_price'].median()

train["categ_chprice"]=train['checkout_price'].apply(pricehighlow,mn=chmean)
train["categ_baseprice"]=train['base_price'].apply(pricehighlow,mn=bsmean)


test["categ_chprice"]=test['checkout_price'].apply(pricehighlow,mn=chmean)
test["categ_baseprice"]=test['base_price'].apply(pricehighlow,mn=bsmean)




train['extrapaid']=np.where((train['checkout_price']-train['base_price'])>0,1,0)
test['extrapaid']=np.where((test['checkout_price']-test['base_price'])>0,1,0)


train["checkout_pricelg"]=np.log(train["checkout_price"])
train["base_pricelg"]=np.log(train["base_price"])
#
test["checkout_pricelg"]=np.log(test["checkout_price"])
test["base_pricelg"]=np.log(test["base_price"])


mealcount=train.groupby("meal_id")["base_price"].count().reset_index()


def fameousmeals(txt):
    if(txt>10000):
        return 4
    elif(txt<10000 and txt>8000):
        return 3
    elif(txt<8000 and txt>5000):
        return 2
    else:
        return 1



mealcount["famousmealscateg"]=mealcount["base_price"].apply(fameousmeals)


train=train.merge(mealcount[["meal_id","famousmealscateg"]],on="meal_id",how='left')
test=test.merge(mealcount[["meal_id","famousmealscateg"]],on="meal_id",how='left')




train=train[train["checkout_price"]>90]
train=train[train["base_price"]>110]

colx=['center_id', 'meal_id',
       'emailer_for_promotion', 'homepage_featured', 'city_code',
       'region_code', 'op_area',
       'center_typech', 'categorych', 'cuisinech','Quarter',
       'year','extrapaid',
       'categ_baseprice','checkout_pricelg',
       'Month','tquarters']


ddt=train[colx].describe()

X_train,X_test, y_train, y_test = train_test_split(
train[colx],train['num_orders2'], test_size=0.33, random_state=42)


clf=lgb.LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
              importance_type='split', learning_rate=0.1, max_depth=-1,
              min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
              n_estimators=2000, n_jobs=-1, num_leaves=31, objective=None,
              random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
              subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

clf.fit(X_train,y_train)
pred=clf.predict(X_test)
mean_squared_error(y_test, pred)

fig, ax = plt.subplots()
ax.scatter(y_test, pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


feat_importances = pd.Series(clf.feature_importances_, index=train[colx].columns)
feat_importances.plot(kind='barh')



clf.fit(train[colx],train['num_orders2'])
pred=clf.predict(test[colx])


test["num_orders"]=pred

test["num_orders"]=round(np.exp(test["num_orders"]))

test["num_orders"].plot()


test[["id","num_orders"]].to_csv(r"D:\24Projects\Food demand Forcast\output.csv",index=False)
