/*
 * Apache License
 * Version 2.0, January 2004
 * http://www.apache.org/licenses/
 * 
 * Copyright 2025 Swapnaneel Dutta
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */



#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H






typedef struct {
double slope;
double intercept;
}LinearRegressionResult;


/*
 * Function: linear_regression_train
 */

void linear_regression_train(LinearRegressionResult *result_set,double *x_data, double *y_data, int number_of_data);
/* ----------------------------
 * This function performs Simple Linear Regression (SLR) to find the best-fit line 
 * through a set of data points. The goal is to determine the line that best models 
 * the relationship between the independent variable (x_data) and the dependent variable (y_data).
 *
 * In Simple Linear Regression, we fit a line to the data of the form:
 *    y = m * x + b
 * where:
 *    - m is the slope (coefficient) of the line, representing the rate of change of y with respect to x
 *    - b is the intercept of the line, representing the value of y when x = 0
 *
 * The function uses the least squares method, which minimizes the sum of squared differences 
 * between the actual data points and the predicted values from the model. The least squares method 
 * provides the best estimates for the slope (m) and intercept (b).
 *
 * The formulae used in the function are based on statistical means:
 *    - m (slope) = (mean(xy) - mean(x) * mean(y)) / (mean(x^2) - mean(x)^2)
 *    - b (intercept) = mean(y) - m * mean(x)
 *
 * Where:
 *    - mean(x) is the average of the independent variable x
 *    - mean(y) is the average of the dependent variable y
 *    - mean(xy) is the average of the product of corresponding x and y values
 *    - mean(x^2) is the average of the square of x values
 *
 * This function calculates the slope and intercept of the best-fit line by:
 * 1. Computing the means of x, y, xy, and x^2.
 * 2. Using the least squares method to calculate the slope (m) and intercept (b).
 *
 * Once computed, the function updates the provided `result_set` structure with the slope and intercept.
 * These values can then be used to predict new y values for given x inputs, or to analyze the linear 
 * relationship between x and y in the dataset.
 *
 * Arguments:
 *    - result_set: A pointer to a `LinearRegressionResult` structure, which will be updated with the
 *      calculated slope and intercept.
 *    - x_data: A pointer to an array of doubles representing the independent variable (x).
 *    - y_data: A pointer to an array of doubles representing the dependent variable (y).
 *    - number_of_data: The number of data points in both x_data and y_data (should be at least 2).
 *
 * Returns:
 *    - This function does not return a value. Instead, it directly updates the `result_set` structure.
 *      After the function completes, the `result_set` will contain:
 *        - result_set->slope: the slope (m) of the best-fit line
 *        - result_set->intercept: the intercept (b) of the best-fit line
 *
 * Note:
 *    - The function **does not allocate memory** for the `result_set` structure. It is the caller's responsibility
 *      to allocate memory for the `LinearRegressionResult` structure and free it after use.
 *    - The function performs no checks on the validity of `x_data` or `y_data` beyond ensuring that the 
 *      `number_of_data` is greater than 1. The caller should ensure valid and matching data arrays are passed in.
 *    - This function assumes that `number_of_data >= 2`. If the data set contains fewer than 2 points, 
 *      the function will not compute any result.
 *
 * Statistical terms used:
 *    - **Mean**: The average of a set of values.
 *    - **Least Squares Method**: A statistical approach used to find the best-fitting line by minimizing 
 *      the sum of squared residuals (differences between observed and predicted values).
 *    - **Residuals**: The differences between observed values and the corresponding predicted values.
 */


/*
 * Function: linear_regression_predict
 */

double linear_regression_predict(LinearRegressionResult result_set, double x_new);
/* ----------------------------
 * This function predicts the dependent variable (y) for a new value of the independent 
 * variable (x_new) based on a trained linear regression model. The model is represented 
 * by the slope (m) and intercept (b), which are the results of a previously performed 
 * linear regression analysis.
 *
 * The function calculates the predicted value of y using the simple linear regression 
 * equation:
 *    y = m * x + b
 *
 * Where:
 *    - m is the slope of the regression line (accessed from result_set.slope)
 *    - b is the intercept of the regression line (accessed from result_set.intercept)
 *    - x is the new value of the independent variable (x_new) for which we want to predict y
 *
 * The function assumes that the `LinearRegressionResult` structure contains the slope (m) 
 * and intercept (b) derived from a previous training step (such as the `linear_regression_train` function).
 *
 * Arguments:
 *    - result_set: A `LinearRegressionResult` structure containing the trained model, 
 *      which includes the slope (m) and intercept (b) values for the regression line.
 *    - x_new: A double representing the new independent variable value (x) for which 
 *      the prediction of y is to be made.
 *
 * Returns:
 *    - This function returns a `double`, which is the predicted value of the dependent 
 *      variable y for the given value of x_new.
 *
 * Example:
 *    LinearRegressionResult result_set = {2.5, 1.0}; // Example model with slope 2.5 and intercept 1.0
 *    double x_new = 10.0;
 *    double y_pred = linear_regression_predict(result_set, x_new); 
 *    // y_pred will be 26.0 since y = 2.5 * 10 + 1.0
 *
 * Notes:
 *    - The function assumes that the `LinearRegressionResult` contains valid, previously 
 *      computed slope (m) and intercept (b) values. If these values are not properly 
 *      set, the function may not produce accurate predictions.
 *    - The function uses simple linear regression and assumes that the relationship 
 *      between the independent and dependent variables is linear.
 *    - This function does not perform any validation of the inputs. It is the caller's 
 *      responsibility to ensure that the `result_set` contains valid regression coefficients 
 *      and that `x_new` is within a reasonable range based on the dataset.
 *    - The function is designed for cases where a linear model has already been trained. 
 *      If the model has not been trained (e.g., `result_set` contains incorrect or default values), 
 *      the predictions may not be meaningful.
 */



#endif // LINEAR_REGRESSION_H
