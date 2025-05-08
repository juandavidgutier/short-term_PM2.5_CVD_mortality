#version modules
#dowhy==0.11.1
#econml==0.15.1


# importing required libraries
import os, warnings, random
import dowhy
import econml
from dowhy import CausalModel
import pandas as pd
import numpy as np
np.int = np.int32
from econml.dml import SparseLinearDML, DML
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from zepid.graphics import EffectMeasurePlot
import matplotlib.pyplot as plt
import scipy.stats as stats
from zepid.graphics import EffectMeasurePlot
from itertools import product
from econml.utilities import WeightedModelWrapper
from plotnine import ggplot, aes, geom_line, geom_ribbon, ggtitle, labs, geom_point, geom_hline, theme_linedraw, theme, element_rect, theme_light, element_line, element_text, geom_errorbarh, geom_vline, theme_bw, element_blank

seed = 42
np.random.seed(seed)
random.seed(seed)


#%%#
#dataframe ATE
# Set display options for pandas to show 5 decimal places
pd.set_option('display.float_format', lambda x: '{:.5f}'.format(x))

# Create data frame of ATE results
df_ATE = pd.DataFrame(0.0, index=range(0, 4), columns=['ATE', '95% CI']).astype({'ATE': 'float64'})

# Convert the second column to tuples with 5 decimal places
df_ATE['95% CI'] = [((0.0, 0.0)) for _ in range(4)]  # Using list comprehension to create tuples

# Display the DataFrame
print(df_ATE)

#%%#
#import data
data_all = pd.read_csv("D:/data.csv") 

#subset
data_all = data_all[data_all['Year'] >= 2015]
data_all = data_all[data_all['top'] <= 10]

#%%
# Binary exposure
data_all['PM25'] = (data_all['PM25'] > 10).astype(int)
data_all['avg2_PM25'] = (data_all['avg2_PM25'] > 10).astype(int)
data_all['avg5_PM25'] = (data_all['avg5_PM25'] > 10).astype(int)
data_all['avg7_PM25'] = (data_all['avg7_PM25'] > 10).astype(int)

#%%
# Binary exposure next day
data_all['lead1_PM25'] = (data_all['lead1_PM25'] > 10).astype(int)

#%%
# Other aerosols in sd units
def convert_zscore(dataframe, columns):
    dataframe_zscore = dataframe.copy()
    dataframe_zscore.iloc[:, columns] = stats.zscore(dataframe.iloc[:, columns], axis=0, nan_policy='omit')
    return dataframe_zscore

columns_to_sd = list(range(17, 40))  
data_all = convert_zscore(data_all, columns_to_sd)

#%%#
## Ignore warnings
warnings.filterwarnings('ignore') 

#%%#
# Effect with exposure of the day
data0 = data_all[['excess', 'PM25',
                  'BC', 'DMS', 'PM', 'OC', 'SO2', 'SO4', 'Temperature', 'lead1_PM25']]

data0 = data0.dropna()

Y = data0.excess.to_numpy()
T = data0.PM25.to_numpy()
W = data0[['BC', 'DMS', 'PM', 'OC', 'SO2', 'SO4', 'Temperature']].to_numpy().reshape(-1, 7)
X = data0[['Temperature', 'lead1_PM25']].to_numpy().reshape(-1, 2)


#Estimation of the effect 
estimate0 = SparseLinearDML(featurizer=PolynomialFeatures(degree=3, include_bias=False),
                            discrete_treatment=True, cv=5, random_state=123)

estimate0 = estimate0.dowhy

# fit the model
estimate0.fit(Y=Y, T=T, X=X, W=W, inference='auto') 

# predict effect for each sample X
estimate0.effect(X)

# ATE
ate0 = estimate0.ate(X) 
print(ate0) 

# confidence interval of ate
ci0 = estimate0.ate_interval(X) 
print(ci0) 

# Set values in the df_ATE
df_ATE.at[0, 'ATE'] = round(ate0, 5)
df_ATE.at[0, '95% CI'] = ci0
print(df_ATE)

#%%#
#Refute tests
#with random common cause
random_PM25_0 = estimate0.refute_estimate(method_name="random_common_cause", random_state=123, num_simulations=10)
print(random_PM25_0)

#with replace a random subset of the data
subset_PM25_0 = estimate0.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.9, num_simulations=10)
print(subset_PM25_0)

#with replace a dummy outcome
dummy_PM25_0 = estimate0.refute_estimate(method_name="dummy_outcome_refuter", num_simulations=10)
print(dummy_PM25_0[0])

#with placebo 
placebo_PM25_0 = estimate0.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=10)
print(placebo_PM25_0)

#%%#
#CATE
#range of X
# Find the maximum and minimum values of Temperature
max_value0 = max(X[:, 0])
min_value0 = min(X[:, 0])
min_X0 = min_value0
max_X0 = max_value0
delta0 = (max_X0 - min_X0) / 100
X_test0 = np.arange(min_X0, max_X0 + delta0 - 0.001, delta0).reshape(-1, 1)

# Find the maximum and minimum values of lead1_PM25
max_value1 = max(X[:, 1])
min_value1 = min(X[:, 1])
min_X1 = min_value1
max_X1 = max_value1
delta1 = (max_X1 - min_X1) / 100
X_test1 = np.arange(min_X1, max_X1 + delta1 - 0.001, delta1).reshape(-1, 1)

X_test = np.concatenate((X_test0, X_test1), axis=1)

estimate0_2 =  SparseLinearDML(featurizer=PolynomialFeatures(degree=3, include_bias=False),
                            discrete_treatment=True, cv=5, random_state=123)

estimate0_2.fit(Y=Y, T=T, X=X, W=W, inference='auto')

treatment_cont_marg = estimate0_2.const_marginal_effect(X_test)
hte_lower2_cons, hte_upper2_cons = estimate0_2.const_marginal_effect_interval(X_test)

X_test = X_test[:, 0].ravel()
treatment_cont_marg = treatment_cont_marg.ravel()

# Reshape to 1-dimensional arrays
hte_lower2_cons = hte_lower2_cons.ravel()
hte_upper2_cons = hte_upper2_cons.ravel()

#Figure 2a
(
ggplot()
  + aes(x=X_test, y=treatment_cont_marg)
  + geom_line()
  + geom_ribbon(aes(ymin = hte_lower2_cons, ymax = hte_upper2_cons), alpha=0.1, fill="blue")
  + labs(x='Temperature', y='Effect of high levels of PM2.5 on excess CVD deaths')
  + geom_hline(yintercept=0, color="red", linetype="dashed")
)

#%%
data2 = data_all[['excess', 'avg2_PM25',
                  'avg2_BC', 'avg2_DMS', 'avg2_PM', 'avg2_OC', 'avg2_SO2', 'avg2_SO4', 'avg2_Temperature', 'lead1_PM25']]

data2 = data2.dropna()

Y = data2.excess.to_numpy()
T = data2.avg2_PM25.to_numpy()
W = data2[['avg2_BC', 'avg2_DMS', 'avg2_PM', 'avg2_OC', 'avg2_SO2', 'avg2_SO4', 'avg2_Temperature']].to_numpy().reshape(-1, 7)
X = data2[['avg2_Temperature', 'lead1_PM25']].to_numpy().reshape(-1, 2)

#Estimation of ATE
estimate2 =  SparseLinearDML(featurizer=PolynomialFeatures(degree=3, include_bias=False),
                            discrete_treatment=True, cv=5, random_state=123)

estimate2 = estimate2.dowhy

# fit the model
estimate2.fit(Y=Y, T=T, X=X, W=W, inference='auto')

# predict effect for each sample X
estimate2.effect(X)

# ate
ate2 = estimate2.ate(X)
print(ate2)

# confidence interval of ate
ci2 = estimate2.ate_interval(X)
print(ci2)

# Set values in the df_ATE
df_ATE.at[1, 'ATE'] = round(ate2, 5)
df_ATE.at[1, '95% CI'] = ci2
print(df_ATE)

#%%
#Refute tests
#with random common cause
random_PM25_2 = estimate2.refute_estimate(method_name="random_common_cause", random_state=123, num_simulations=10)
print(random_PM25_2)

#with replace a random subset of the data
subset_PM25_2 = estimate2.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.9, num_simulations=10)
print(subset_PM25_2)

#with replace a dummy outcome
dummy_PM25_2  = estimate2.refute_estimate(method_name="dummy_outcome_refuter", num_simulations=10)
print(dummy_PM25_2 [0])

#with placebo
placebo_PM25_2 = estimate2.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=10)
print(placebo_PM25_2)

#%%
#CATE
#range of X
# Find the maximum and minimum values of avg2_Temperature
max_value0 = max(X[:, 0])
min_value0 = min(X[:, 0])
min_X0 = min_value0
max_X0 = max_value0
delta0 = (max_X0 - min_X0) / 100
X_test0 = np.arange(min_X0, max_X0 + delta0 - 0.001, delta0).reshape(-1, 1)

# Find the maximum and minimum values of lead1_PM25
max_value1 = max(X[:, 1])
min_value1 = min(X[:, 1])
min_X1 = min_value1
max_X1 = max_value1
delta1 = (max_X1 - min_X1) / 100
X_test1 = np.arange(min_X1, max_X1 + delta1 - 0.001, delta1).reshape(-1, 1)

X_test = np.concatenate((X_test0, X_test1), axis=1)

estimate2_2 =  SparseLinearDML(featurizer=PolynomialFeatures(degree=3, include_bias=False),
                            discrete_treatment=True, cv=5, random_state=123)

estimate2_2.fit(Y=Y, T=T, X=X, W=W, inference='auto')

treatment_cont_marg = estimate2_2.const_marginal_effect(X_test)
hte_lower2_cons, hte_upper2_cons = estimate2_2.const_marginal_effect_interval(X_test)

# Reshape X_test to 1-dimensional
X_test = X_test[:, 0].ravel()
treatment_cont_marg = treatment_cont_marg.ravel()

# Reshape to 1-dimensional arrays
hte_lower2_cons = hte_lower2_cons.ravel()
hte_upper2_cons = hte_upper2_cons.ravel()

#Figure 2b
(
ggplot()
  + aes(x=X_test, y=treatment_cont_marg)
  + geom_line()
  + geom_ribbon(aes(ymin = hte_lower2_cons, ymax = hte_upper2_cons), alpha=0.1, fill="blue")
  + labs(x='Temperature', y='Effect of high levels of PM2.5 on excess CVD deaths')
  + geom_hline(yintercept=0, color="red", linetype="dashed")
)

#%%
data5 = data_all[['excess', 'avg5_PM25',
                  'avg5_BC', 'avg5_DMS', 'avg5_PM', 'avg5_OC', 'avg5_SO2', 'avg5_SO4', 'avg5_Temperature', 'lead1_PM25']]

data5 = data5.dropna()

Y = data5.excess.to_numpy()
T = data5.avg5_PM25.to_numpy()
W = data5[['avg5_BC', 'avg5_DMS', 'avg5_PM', 'avg5_OC', 'avg5_SO2', 'avg5_SO4', 'avg5_Temperature']].to_numpy().reshape(-1, 7)
X = data5[['avg5_Temperature', 'lead1_PM25']].to_numpy().reshape(-1, 2)

#Estimation of ATE
estimate5 =  SparseLinearDML(featurizer=PolynomialFeatures(degree=3, include_bias=False),
                            discrete_treatment=True, cv=5, random_state=123)

estimate5 = estimate5.dowhy

# fit the model
estimate5.fit(Y=Y, T=T, X=X, W=W, inference='auto')

# predict effect for each sample X
estimate5.effect(X)

# ate
ate5 = estimate5.ate(X)
print(ate5)

# confidence interval of ate
ci5 = estimate5.ate_interval(X)
print(ci5)

# Set values in the df_ATE
df_ATE.at[2, 'ATE'] = round(ate5, 5)
df_ATE.at[2, '95% CI'] = ci5
print(df_ATE)

#%%
#Refute tests
#with random common cause
random_PM25_5 = estimate5.refute_estimate(method_name="random_common_cause", random_state=123, num_simulations=10)
print(random_PM25_5)

#with replace a random subset of the data
subset_PM25_5 = estimate5.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.9, num_simulations=10)
print(subset_PM25_5)

#with replace a dummy outcome
dummy_PM25_5  = estimate5.refute_estimate(method_name="dummy_outcome_refuter", num_simulations=10)
print(dummy_PM25_5 [0])

#with placebo
placebo_PM25_5 = estimate5.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=10)
print(placebo_PM25_5)

#%%
#CATE
#range of X
# Find the maximum and minimum values of avg5_Temperature
max_value0 = max(X[:, 0])
min_value0 = min(X[:, 0])
min_X0 = min_value0
max_X0 = max_value0
delta0 = (max_X0 - min_X0) / 100
X_test0 = np.arange(min_X0, max_X0 + delta0 - 0.001, delta0).reshape(-1, 1)

# Find the maximum and minimum values of lead1_PM25
max_value1 = max(X[:, 1])
min_value1 = min(X[:, 1])
min_X1 = min_value1
max_X1 = max_value1
delta1 = (max_X1 - min_X1) / 100
X_test1 = np.arange(min_X1, max_X1 + delta1 - 0.001, delta1).reshape(-1, 1)

X_test = np.concatenate((X_test0, X_test1), axis=1)

estimate5_2 =  SparseLinearDML(featurizer=PolynomialFeatures(degree=3, include_bias=False),
                            discrete_treatment=True, cv=5, random_state=123)

estimate5_2.fit(Y=Y, T=T, X=X, W=W, inference='auto')

treatment_cont_marg = estimate5_2.const_marginal_effect(X_test)
hte_lower2_cons, hte_upper2_cons = estimate5_2.const_marginal_effect_interval(X_test)

# Reshape X_test to 1-dimensional
X_test = X_test[:, 0].ravel()
treatment_cont_marg = treatment_cont_marg.ravel()

# Reshape to 1-dimensional arrays
hte_lower2_cons = hte_lower2_cons.ravel()
hte_upper2_cons = hte_upper2_cons.ravel()

#Figure 2c
(
ggplot()
  + aes(x=X_test, y=treatment_cont_marg)
  + geom_line()
  + geom_ribbon(aes(ymin = hte_lower2_cons, ymax = hte_upper2_cons), alpha=0.1, fill="blue")
  + labs(x='Temperature', y='Effect of high levels of PM2.5 on excess CVD deaths')
  + geom_hline(yintercept=0, color="red", linetype="dashed")
)

#%%
data7 = data_all[['excess', 'avg7_PM25',
                  'avg7_BC', 'avg7_DMS', 'avg7_PM', 'avg7_OC', 'avg7_SO2', 'avg7_SO4', 'avg7_Temperature', 'lead1_PM25']]

data7 = data7.dropna()

Y = data7.excess.to_numpy()
T = data7.avg7_PM25.to_numpy()
W = data7[['avg7_BC', 'avg7_DMS', 'avg7_PM', 'avg7_OC', 'avg7_SO2', 'avg7_SO4', 'avg7_Temperature']].to_numpy().reshape(-1, 7)
X = data7[['avg7_Temperature', 'lead1_PM25']].to_numpy().reshape(-1, 2)

#Estimation of ATE
estimate7 =  SparseLinearDML(featurizer=PolynomialFeatures(degree=3, include_bias=False),
                            discrete_treatment=True, cv=5, random_state=123)

estimate7 = estimate7.dowhy

# fit the model
estimate7.fit(Y=Y, T=T, X=X, W=W, inference='auto')

# predict effect for each sample X
estimate7.effect(X)

# ate
ate7 = estimate7.ate(X)
print(ate7)

# confidence interval of ate
ci7 = estimate7.ate_interval(X)
print(ci7)

# Set values in the df_ATE
df_ATE.at[3, 'ATE'] = round(ate7, 5)
df_ATE.at[3, '95% CI'] = ci7
print(df_ATE)

#%%
#Refute tests
#with random common cause
random_PM25_7 = estimate7.refute_estimate(method_name="random_common_cause", random_state=123, num_simulations=10)
print(random_PM25_7)

#with replace a random subset of the data
subset_PM25_7 = estimate7.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.9, num_simulations=10)
print(subset_PM25_7)

#with replace a dummy outcome
dummy_PM25_7  = estimate7.refute_estimate(method_name="dummy_outcome_refuter", num_simulations=10)
print(dummy_PM25_7 [0])

#with placebo
placebo_PM25_7 = estimate7.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=10)
print(placebo_PM25_7)

#%%
#CATE
#range of X
# Find the maximum and minimum values of avg7_Temperature
max_value0 = max(X[:, 0])
min_value0 = min(X[:, 0])
min_X0 = min_value0
max_X0 = max_value0
delta0 = (max_X0 - min_X0) / 100
X_test0 = np.arange(min_X0, max_X0 + delta0 - 0.001, delta0).reshape(-1, 1)

# Find the maximum and minimum values of lead1_PM25
max_value1 = max(X[:, 1])
min_value1 = min(X[:, 1])
min_X1 = min_value1
max_X1 = max_value1
delta1 = (max_X1 - min_X1) / 100
X_test1 = np.arange(min_X1, max_X1 + delta1 - 0.001, delta1).reshape(-1, 1)

X_test = np.concatenate((X_test0, X_test1), axis=1)

estimate7_2 =  SparseLinearDML(featurizer=PolynomialFeatures(degree=3, include_bias=False),
                            discrete_treatment=True, cv=5, random_state=123)

estimate7_2.fit(Y=Y, T=T, X=X, W=W, inference='auto')

treatment_cont_marg = estimate7_2.const_marginal_effect(X_test)
hte_lower2_cons, hte_upper2_cons = estimate7_2.const_marginal_effect_interval(X_test)

# Reshape X_test to 1-dimensional
X_test = X_test[:, 0].ravel()
treatment_cont_marg = treatment_cont_marg.ravel()

# Reshape to 1-dimensional arrays
hte_lower2_cons = hte_lower2_cons.ravel()
hte_upper2_cons = hte_upper2_cons.ravel()

#Figure 2d
(
ggplot()
  + aes(x=X_test, y=treatment_cont_marg)
  + geom_line()
  + geom_ribbon(aes(ymin = hte_lower2_cons, ymax = hte_upper2_cons), alpha=0.1, fill="blue")
  + labs(x='Temperature', y='Effect of high levels of PM2.5 on excess CVD deaths')
  + geom_hline(yintercept=0, color="red", linetype="dashed")
)

#%%
#Figure 1
labs = ['PM2.5',
        'PM2.5 avg2',
        'PM2.5 avg5',
        'PM2.5 avg7' ]

df_labs = pd.DataFrame({'Labels': labs})

print(df_ATE)

# Convert ATE to separate DataFrame
ATE = df_ATE[['ATE']].round(5)
print(ATE)

# Convert tuples in the '95% CI' column to separate DataFrame
df_ci = df_ATE['95% CI'].apply(pd.Series)

# Rename columns in df_ci
df_ci.columns = ['Lower', 'Upper']

# Create two separate DataFrames for Lower and Upper
Lower = df_ci[['Lower']].copy()
print(Lower)
Upper = df_ci[['Upper']].copy()
print(Upper)

df_plot = pd.concat([df_labs.reset_index(drop=True), ATE, Lower, Upper], axis=1)
print(df_plot)

p = EffectMeasurePlot(label=df_plot.Labels, effect_measure=df_plot.ATE, lcl=df_plot.Lower, ucl=df_plot.Upper)
p.labels(center=0)
p.colors(pointcolor='r' , pointshape="s", linecolor='b')
p.labels(effectmeasure='ATE')  
p.plot(figsize=(10, 5), t_adjuster=0.10, max_value=1.0, min_value=-1.0)
plt.tight_layout()
plt.show()
