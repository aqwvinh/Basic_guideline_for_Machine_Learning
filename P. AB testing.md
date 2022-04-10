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

### Chose the right statistical test
Need to know:
- whether the data meets certain assumptions: **independence of observations**, same variance within each group, normality of data. If your data do not meet the assumptions of **normality or homogeneity of variance**, you may be able to perform a **nonparametric statistical test**, which allows you to make comparisons without any assumptions about the data distribution.
- types of variables: numerical (continuous or discrete) or categorical (ordinal, nominal, binary)

**Parametric tests** usually have stricter requirements than **nonparametric test**s, and are able to make stronger inferences from the data.

Tests: regression, comparison or correlation

**Parametric tests**
Comparison:
- T-tests are used when comparing the means of precisely two groups (e.g. the average heights of men and women)
- ANOVA and MANOVA tests are used when comparing the means of more than two groups (e.g. the average heights of children, teenagers, and adults).

Correlation:
- Pearson

**Nonparametric tests**
Non-parametric tests don’t make as many assumptions about the data, and are useful when one or more of the common statistical assumptions are violated. <br>However, the inferences they make aren’t as strong as with parametric tests.

Spearman, Chi-2 etc



## 5. Drawing conclusions

Since our p-value=0.732 is way above our α=0.05 threshold, we cannot reject the Null hypothesis Hₒ, which means that our new design did not perform significantly different than the old one

Additionally, if we look at the confidence interval for the treatment group `([0.116, 0.135]`, or 11.6-13.5%) we notice that:
- It includes our baseline value of 13% conversion rate
- It does not include our target value of 15% (the 2% uplift we were aiming for)

What this means is that it is more likely that the true conversion rate of the new design is similar to our baseline, rather than the 15% target we had hoped for. This is further proof that our new design is not likely to be an improvement on our old design


## Bonus

- Multivariate testing: more changes than A/B testing. More powerful but more difficult to implement and need more data
- What to do if A/B test fails? ==> check H0, increase sample size/duration of test, check data measurements, check data quality

### Causal inference
Important cuz ML is good at finding correlation in data but not causation. Big data is not the solution cuz don't find causation.Causation: cause ==> effect, the cause is partly responsible for the effect. 
<br>Correlation is different than causation. Ex: correlation between the consumption of chocolate and the nb of Nobel Prizes per country: a strong correlation but not causality. 
<br>Applications: determine factors that impact the decision ==> better allocate resources and/or rank the impact of different factors on purchasing decision for ex. 
<br>Different methods to analyse causality:
- Monte Carlo Simulation: run a large nb of simulations within a model and look at the outputs of each simulation. Two types of inputs: **certain inputs and uncertain inputs**. Same value for certain inputs and a value is assigned to all uncertain inputs according to its **probability distribution**. 
- Markov chain: every node in a BN is conditionally **independent of its nondescendants**, given its parents. A node has no effect on nodes which do not descend from it. 
- **Bayesian network (DAG)**: Visual map showing **all variances that influence each othe, through nodes and edgesr**. It's bayesian because the edges can be valued by the **probability of the child node given the parent node**.
- - How to determine the direction of causality ==> **hold a node constant and then observe the effect**
- - Bayes theorem: P(A|B)= P(B|A).P(A)/P(B). The equation consists of four parts; **the posterior probability** is the probability that B occurs given A (proba of the hypothesis A given the osbserved evidence B). **The conditional probability or likelihood** is P(B|A), the probability of the evidence given that the hypothesis A is true. This can be derived from the data. **The prior** P(A) is the probability of the hypothesis before observing the evidence. This can also be derived from the data or domain knowledge. The prior is important cuz it determines how strongly we weight the likelihood. Finally, **the marginal probability** P(B) describes the probability of the new evidence under all possible hypotheses which needs to be computed.
- - There are two methods to build the right DAG: **score-based structure learning** and **constraint-based**
- - Steps for a BN: build structure (automatically from data or expert knowledge) ==> review structure (check each relationship) ==> likelihood estimation (quantify the edges by observational data) ==> prediction & inference (use the structure and the likelihoods to make predictions)
- - Advatanges: allow the introduction of business rules (select variables, explanation for counterintuitive results), easily interpretable, statiscally significant (learned from the data). 
- - Limitations: require lots of data to capture all possible variables + need to have a search strategy to build the best DAG (lots of possibilities)
- - Library: ```Causalnex```==> powerful cuz can automatically build the structure and add/rm relationships by expert or ```bnlearn```
- SEM (Structural Equation Modeling): flexible framework that can measure the relationships variables. Powerful because it manages the measurement error ==> enable the development and the analysis of complex relationships among multiple variables


### Parameter estimation
Data collected in the real world is almost never representative of the entire population ==> **need to estimate distribution parameters from an observed sample population**
<br> Assumption: i.i.d ==> all data samples are considered independent
There are two main methods:
- **Maximum Likelihood Estimation (MLE)**: frequentist view (based on observational experience then law of large numbers to generalize the observed results on a sample to the whole population). Often use log MLE because it's easier to take derivatives of sums than products. For a Gaussian random variable ==> the MLE solution is simply the mean and the variance of the observed data. **Find parameters (theta, lambda etc)**. Prediction: P(X|D) = P(X|theta)
- **Bayesian Estimation**: bayesian view (based on prior knowledge). Use previous studies to have some **prior knowledge about the distribution**. The equation used for Bayesian estimation takes on the same form as Bayes’ theorem, the key difference being that we are now using models and probability density functions (pdfs) in place of numerical probabilities. **Find distribution (density functions)**. P(X|D) is the integral of density functions and the random variable theta ==> it's more complex (use both the posterior distribution and the distribution over the random variable theta)

When to use MLE or Bayesian estimation?
- How much data? MLE needs more data
- Do you have reliable prior knowledge?
- Bayesian estimation is computational more complex, MLE is simplier to implement
