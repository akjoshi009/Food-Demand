import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn import preprocessing




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

colx=['week', 'center_id', 'meal_id',
       'emailer_for_promotion', 'homepage_featured', 'city_code',
       'region_code', 'op_area',
       'center_typech', 'categorych', 'cuisinech', 'extrapaid',
       'checkout_pricelg', 'base_pricelg']


train['extrapaid']=np.where((train['checkout_price']-train['base_price'])>0,1,0)
test['extrapaid']=np.where((test['checkout_price']-test['base_price'])>0,1,0)


ddt=train[colx].describe()

train["checkout_pricelg"]=np.log(train["checkout_price"])
train["base_pricelg"]=np.log(train["base_price"])

test["checkout_pricelg"]=np.log(test["checkout_price"])
test["base_pricelg"]=np.log(test["base_price"])


train["meal_id"].value_counts()

train=train[train["checkout_price"]>70]
train=train[train["base_price"]>90]




X_train,X_test, y_train, y_test = train_test_split(
train[colx],train['num_orders2'], test_size=0.33, random_state=42)


clf=lgb.LGBMRegressor(boosting_type='gbdt', objective='regression', num_leaves=1200,
                                learning_rate=0.17, n_estimators=500, max_depth=25,
                                metric='rmse', bagging_fraction=0.8, feature_fraction=0.8, reg_lambda=0.9)
clf.fit(X_train,y_train)
pred=clf.predict(X_test)
mean_squared_error(y_test, pred)


feat_importances = pd.Series(clf.feature_importances_, index=train[colx].columns)
feat_importances.plot(kind='barh')



clf.fit(train[colx],train['num_orders2'])
pred=clf.predict(test[colx])

test["num_orders"]=pred

test["num_orders"]=round(np.exp(test["num_orders"]))


test[["id","num_orders"]].to_csv(r"D:\24Projects\Food demand Forcast\output.csv",index=False)
