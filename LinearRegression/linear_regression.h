#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H






typedef struct {
double slope;
double intercept;
}LinearRegressionResult;


/*
 * Function: linear_regression
 */

void linear_regression(LinearRegressionResult *result_set,double *x_data, double *y_data, int number_of_data);
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

#endif // LINEAR_REGRESSION_H
