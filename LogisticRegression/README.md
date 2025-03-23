# Logistic Regression: Mathematical Explanation

Logistic regression is a statistical method used for binary classification. It predicts the probability that an input belongs to a certain class.

## 1. logistic Model (Prediction)

The first step in logistic regression is using a **logistic model** to predict the output. Given a set of input features \( X = [x_1, x_2, ..., x_n] \) and corresponding weights \( \theta = [\theta_0, \theta_1, ..., \theta_n] \), the model computes a logistic combination:

\[
z = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n = \theta^T X
\]

Where:
- \( z \) is the raw output (which can be any real number, positive or negative),
- \( \theta \) represents the weights (parameters) of the model,
- \( X \) represents the input features.

## 2. Sigmoid Function

The raw output \( z \) from the logistic model is then passed through a **sigmoid function** (also known as the logistic function), which squashes the output between 0 and 1. This is what makes logistic regression a probabilistic classifier, where the output is interpreted as the probability that an instance belongs to a certain class.

The **sigmoid function** is defined as:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Where:
- \( \sigma(z) \) is the probability that the instance belongs to class 1 (in binary classification),
- \( e^{-z} \) is the exponential function applied to \( -z \).

## 3. Model Output (Probability)

So, the output of logistic regression is:

\[
p(y = 1 | X) = \sigma(\theta^T X) = \frac{1}{1 + e^{-(\theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n)}}
\]

This represents the **probability** that the instance \( X \) belongs to the positive class (class 1). The complement \( 1 - p(y = 1 | X) \) is the probability that the instance belongs to the negative class (class 0).

## 4. Cost Function (Log-Loss)

To train the model, we need to **optimize** the parameters \( \theta \). We do this by minimizing the **cost function**, which is based on the **log-likelihood** of the predicted probabilities.

The **log-likelihood** function for logistic regression is:

\[
L(\theta) = \sum_{i=1}^{m} \left[ y^{(i)} \log(p^{(i)}) + (1 - y^{(i)}) \log(1 - p^{(i)}) \right]
\]

Where:
- \( m \) is the number of training examples,
- \( y^{(i)} \) is the actual label for the \( i \)-th training example,
- \( p^{(i)} = \sigma(\theta^T X^{(i)}) \) is the predicted probability for the \( i \)-th training example.

We minimize the negative of this function during model training.

## 5. Gradient Descent (Optimization)

The parameters \( \theta \) are optimized using **gradient descent**. The goal is to minimize the cost function by iteratively updating the parameters in the direction of the steepest descent.

The gradient of the log-likelihood with respect to \( \theta_j \) is:

\[
\frac{\partial L(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left[ p^{(i)} - y^{(i)} \right] x_j^{(i)}
\]

Where:
- \( x_j^{(i)} \) is the \( j \)-th feature of the \( i \)-th training example.

The gradient descent update rule is then:

\[
\theta_j := \theta_j - \alpha \frac{\partial L(\theta)}{\partial \theta_j}
\]

Where:
- \( \alpha \) is the learning rate (a hyperparameter that controls the step size during each iteration).

## 6. Decision Boundary

Once the model is trained, we determine the decision boundary. This is the threshold at which we classify an instance as belonging to class 1 or class 0.

The decision rule is based on the output probability:

\[
p(y = 1 | X) \geq 0.5 \Rightarrow \hat{y} = 1
\]
\[
p(y = 1 | X) < 0.5 \Rightarrow \hat{y} = 0
\]

Thus, if the predicted probability is greater than or equal to 0.5, we classify the instance as class 1; otherwise, it is classified as class 0.

---

This is the basic mathematical framework behind logistic regression, allowing it to make predictions about probabilities and classify instances into different classes based on the learned weights and features.

### Our Implementation

#### Training
1. We start with \( X \), \( Y \), \(\alpha \) (pre defined learning constant) from training set and a randomised \( \theta \)
2. We compute \( Z  =  X \theta \)
3. We make predictions set \( Ynew =  \sigma (z_i)\)
4. We calculate gradient \( grad = X^T(Ynew - Y)\)
5. We update the weights \( \theta := \theta- \frac{\alpha}{m}gradient \)
6. We repeat the steps for a pre defined number of iterations.

#### Prediction 
Once we have optimised the weights with a certain number of iterations, we can simply predict new values by:

\[
Zpredicted = Xnew \theta
\]

\[
Ypredicted_i = \sigma(Zpredicted_i)
\]
Now we can get the binary values from the decision bounary probability function for the values in \(Ypredicted\) vector.

---
### :file_folder: Files
1. **logistic_regression.c** (Implementation source)
2. **logistic_regression.h** (Header that contains definations and usage guide)
3. **usage.c** (A basic main containing file showing the usage of the function)
---
### :gear: Usage and Testing
To use the function you need to include the header file and compile it with the source.
You may look at the documentation in **logistic_regression.h** or check out the basic usage in **usage.c**
##### Compilation
```
gcc -o your_program your_program.c logistic_regression.c -lgsl -lm
```
To simply test the usage with **usage.c** you may compile it as:
```
gcc -o usage usage.c logistic_regression.c -lgsl -lm
```
