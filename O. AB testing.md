# A/B testing

Steps:
1. Designing experiment
2. Collecting and preparing the data
3. Visualising the results
4. Testing the hypothesis
5. Drawing conclusions

## 1. Designing experiment

A. Formulate the **hypothesis**:
- One-tail test ==> change in one direction, good to measure the impact of a chance but not testing the effect in the other direction
- Two-tailed test ==> change in both direction, worse or better, we don't know in advance

Let's say p and p0 new and old metric (such as conversion rate)
<br>H0: p = p0 ==> no effect
<br>H: p ≠ p0 ==> so < or >

B. Choose a **confidence level**, usually alpha=5% for a confidence of 95%

C. Choose the **variables**: 

*Independent variable*
<br>usually split in two groups:
- A ```control```group ==> They will be shown the old parameter/design 
- A ```treatment/experimental```group ==> show the new

*Dependent variable*
<br>It's what we are trying to measure. Example: for the conversion rate, we can set 1 if the user buys and 0 if not
<br>Then it's easy to calculate the mean of each group

D. Choose a **sample size**

We can use the *power analysis* to calculate the sample size, using python
<br>We have to determine in advance:
- Power of the test (probability of finding a statistical difference when there is actually one - usually set to 0.8)
- Confidence: alpha = 5%
- Effect size: how big of a difference we expect from the results? In this example, we can use 13% (old) and 15% (target)


```
# Packages imports
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.api as sms
from math import ceil

# Calculating effect size based on our expected rates
effect_size = sms.proportion_effectsize(0.13, 0.15)    

# Calculating sample size needed
required_n = sms.NormalIndPower().solve_power(
    effect_size, 
    power=0.8, 
    alpha=0.05, 
    ratio=1
    )                                                  
# Rounding up to next whole number                          
required_n = ceil(required_n)                          
# We need required_n observations for each group
print(required_n)
```

## 2. Collecting and preparing data

Implementation of the test (with the engineering team). Make sure to categorize **control** and **experimental** group, **old** and **new** parameter and the **result**

## 3. Visualising the results

Compute the mean, std and variance of each group
<br>Always nice to plot the results

## 4. Testing the hypothesis

If we have a large sample size, we can use the **normal approximation** for calculating the **p-value** (ie Z-test)
<br>Again, we can use python and the library `statsmodels`


```
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

target = "converted"
control_results = df[df["group"] == "control"][target]
treatment_results = df[df["group"] == "treatment"][target]

n_con = control_results.count()
n_treat = treatment_results.count()
successes = [control_results.sum(), treatment_results.sum()]
nobs = [n_con, n_treat]

z_stat, pval = proportions_ztest(successes, nobs=nobs)
(lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(successes, nobs=nobs, alpha=0.05)

print(f'z statistic: {z_stat:.2f}')
print(f'p-value: {pval:.3f}')
print(f'Confidence Interval 95% for control group: [{lower_con:.3f}, {upper_con:.3f}]')
print(f'Confidence Interval 95% for treatment group: [{lower_treat:.3f}, {upper_treat:.3f}]')
```

We got the following results

```
z statistic: -0.34
p-value: 0.732
ci 95% for control group: [0.114, 0.133]
ci 95% for treatment group: [0.116, 0.135]
```


## 5. Drawing conclusions

Since our p-value=0.732 is way above our α=0.05 threshold, we cannot reject the Null hypothesis Hₒ, which means that our new design did not perform significantly different than the old one

Additionally, if we look at the confidence interval for the treatment group `([0.116, 0.135]`, or 11.6-13.5%) we notice that:
- It includes our baseline value of 13% conversion rate
- It does not include our target value of 15% (the 2% uplift we were aiming for)

What this means is that it is more likely that the true conversion rate of the new design is similar to our baseline, rather than the 15% target we had hoped for. This is further proof that our new design is not likely to be an improvement on our old design




