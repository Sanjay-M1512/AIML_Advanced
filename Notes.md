## **Advance Machine Learning Techniques**
- **There are 5 modules in this percular training**

## **A. MODULE 1**
- ### **Supervised Machine Learning**
1. Naive Bayes Classification
2. Support Vector Machines
3. Decision Trees for Classification

## 1. **Naive Bayes Classification:**
- **NB is Supervised Probabilistic Classification Algorithm based on Bayes Theorem.**

> ***P(C | X) = (P(X | C) * P(C)) / P(X)***
**C - Class(Hypothesis)
X - Input Features(Evidence)
P(C) - Prior probability of class
P(X∣C) - Likelihood
P(C∣X) - Posterior probability (final result)**

> ***C * = argmax C***
**C -> Collection -> C=[P(C_1|X), P(C_2|X), P(C_3|X)] **

1. **Likelihood P(X/C):**
Likelihood is the probability of observing the features of a given class.

**Example:**
- P(Free/Spam)=0.8
- P(Free/Spam)=0.2 | P(Free/Spam)=0.15

2. **Prior Probability:**
Prior Probability is the probability of class before seeing any features.

**Example: 1000emails**
- 600 are spams
- 400 are not spams
> **P(Spam)=600/1000**
> **P(Not Spam)=400/1000**

#### Real World Example:              
1. **Spam Detection(Discrete)**
- **Spam**
- **Not spam**

2. **Weather Data(Continuous)**
- **Temperatue**
- **Wind**

#### **TYPES OF NAIVE BAYES:**
1. **Gaussian**
Continuous
Normal Distribution
Uses: Medical Sector

2. **Multiomodal**
Discrete Counts, TEXT, NLP
Multinomial

3. **Bernoulli's**
Binary Classification -> 0,1

#### **Why Bayes Theorem?**
