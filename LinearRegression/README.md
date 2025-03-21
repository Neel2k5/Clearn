# Linear Regression 
---
[Linear regression](https://en.wikipedia.org/wiki/Simple_linear_regression) is a statistical method used to model the relationship between a dependent variable (also called the response variable) and one or more independent variables (also called predictor variables) by fitting a linear equation (a straight line) to observed data.

We used simple linear regression in this scenario.
### Simple Linear regression
In simple linear regression, the relationship between the independent variable xx and the dependent variable yy is modeled by a straight line:
$$ y = mx + b $$
Where:
- y is the dependent variable (the value you are trying to predict).
- x is the independent variable (the value you are using to make the prediction).
- m is the slope of the line, indicating how much y changes for a one-unit change in x.
- b is the intercept, the value of y when x=0. 
--- 
To find the slope \( m \) and the intercept \( b \) of the best-fit line in linear regression, we use the following formulas:

### 1. Formula for the Slope \( m \)

The slope \( m \) is given by the formula:

$$
m = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}
$$

Where:
- \( x_i \) and \( y_i \) are the individual data points
- \( \bar{x} \) is the mean of the \( x \)-values
- \( \bar{y} \) is the mean of the \( y \)-values

### 2. Formula for the Intercept \( b \)

The intercept \( b \) is calculated using the formula:

$$
b = \bar{y} - m \bar{x}
$$

Where:
- \( \bar{x} \) is the mean of the \( x \)-values
- \( \bar{y} \) is the mean of the \( y \)-values
--- 
### Steps for Prediction

Given a new value \( x_{\text{new}} \), you can predict the corresponding value \( y_{\text{new}} \) by plugging it into the regression equation:

$$
y_{\text{new}} = m \cdot x_{\text{new}} + b
$$

### Example:

Let's say we have:
- **Slope** \( m = 2.5 \)
- **Intercept** \( b = 3 \)

For a new input value \( x_{\text{new}} = 4 \), the predicted \( y_{\text{new}} \) would be:

$$
y_{\text{new}} = 2.5 \cdot 4 + 3 = 10 + 3 = 13
$$

So, the predicted value of \( y \) when \( x = 4 \) is \( y = 13 \).

---
### :file_folder: Files
1. **linear_regression.c** (Implementation source)
2. **linear_regression.h** (Header that contains definations and usage guide)
3. **usage.c** (A basic main containing file showing the usage of the function)
---
### :gear: Usage and Testing
To use the function you need to include the header file and compile it with the source.
You may look at the documentation in **linear_regression.h** or check out the basic usage in **usage.c**
##### Compilation
```
gcc -o your_program your_program.c linear_regression.c
```
To simply test the usage with **usage.c** you may compile it as:
```
gcc -o usage usage.c linear_regression.c
```