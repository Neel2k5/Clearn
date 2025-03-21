# Multiple Linear Regression 
---
[Multiple Linear regression](https://en.wikipedia.org/wiki/Linear_regression) is a mathematical model where we want to model the relationship between a dependent variable **Y** and multiple independent variables **X1​,X2​,…,Xp**​. The equation for MLR with pp predictors is: 
$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p + \epsilon
$$

Where:

- Y is the dependent variable (target you're predicting).
- X1​,X2​,…,Xp​ are the independent variables (features).
- β0​ is the intercept term (where the line intersects the Y-axis).
- β1​,β2​,…,βp​ are the coefficients that determine how each feature affects the output YY.
- ϵ is the error term (representing the difference between the predicted and actual value).

### Matrix Representation of MLR equation
We want to express this equation in matrix form for computational efficiency, especially when working with many variables (dimensions).
$$
Y = X \cdot \beta + \epsilon
$$

1. **Design Matrix \( X \)** (size \( n \times (p+1) \)):

$$
X = \begin{bmatrix}
1 & X_{11} & X_{12} & \dots & X_{1p} \\
1 & X_{21} & X_{22} & \dots & X_{2p} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & X_{n1} & X_{n2} & \dots & X_{np}
\end{bmatrix}
$$

2. **Target Vector \( Y \)** (size \( n \times 1 \)):

$$
Y = \begin{bmatrix}
Y_1 \\
Y_2 \\
\vdots \\
Y_n
\end{bmatrix}
$$

3. **Coefficient Vector \( \beta \)** (size \( (p+1) \times 1 \)):

$$
\beta = \begin{bmatrix}
\beta_0 \\
\beta_1 \\
\vdots \\
\beta_p
\end{bmatrix}
$$

4. **Error Vector \( \epsilon \)** (size \( n \times 1 \)):

$$
\epsilon = \begin{bmatrix}
\epsilon_1 \\
\epsilon_2 \\
\vdots \\
\epsilon_n
\end{bmatrix}
$$

### Solving for Coefficients Using the Normal Equation

We use the **Normal Equation** to solve for the coefficient vector \( \beta \) in the equation \( Y = X \cdot \beta + \epsilon \):

$$
\beta = (X^T X)^{-1} X^T Y
$$

Where:
- \( X^T \) is the transpose of the matrix \( X \),
- \( (X^T X)^{-1} \) is the inverse of \( X^T X \),
- \( X^T Y \) is the matrix multiplication of \( X^T \) and \( Y \).

### After Solving for \( \boldsymbol{\beta} \)

Once you have the **coefficient vector** \( \boldsymbol{\beta} \), we can proceed with two main steps: making predictions and evaluating your model.
### 1. Making Predictions

After obtaining the coefficient vector \( \boldsymbol{\beta} \), you can use it to make predictions for new data.

#### Prediction Formula:

The predicted output \( \hat{Y} \) for new input data \( X_{\text{new}} \) is calculated as:

$$
\hat{Y} = X_{\text{new}} \cdot \boldsymbol{\beta}
$$

Where:
- \( X_{\text{new}} \) is the new input data matrix (including a column of 1’s for the intercept term),
- \( \boldsymbol{\beta} \) is the coefficient vector.


---
### Our Implementation
\(\ X \) and \(\ Y \) are the datasets we will use to train the model and obtain \(\beta \).
Once \(\beta \) is computed from the provided dataset, we may use it to predict values of new inputs by the above equation

---
### :file_folder: Files
1. **multiple_linear_regression.c** (Implementation source)
2. **multiple_linear_regression.h** (Header that contains definations and usage guide)
3. **usage.c** (A basic main containing file showing the usage of the function)
---
### :gear: Usage and Testing
To use the function you need to include the header file and compile it with the source.
You may look at the documentation in **multiple_linear_regression.h** or check out the basic usage in **usage.c**
##### Compilation
```
gcc -o your_program your_program.c multiple_linear_regression.c -lgsl
```
To simply test the usage with **usage.c** you may compile it as:
```
gcc -o usage usage.c multiple_linear_regression.c -lgsl
```