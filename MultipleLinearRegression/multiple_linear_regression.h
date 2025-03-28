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

#ifndef MLR_H
#define MLR_H

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>


/*
 * Function: mlr_train
 * ----------------------------
 */

unsigned int mlr_train(gsl_matrix *x_data, gsl_vector *y_data, gsl_vector *coeff_set, double regularisation_param);
/* This function performs **Multiple Linear Regression (MLR)** using the least squares method 
 * and ridge regularization to compute the coefficients (including the intercept) for the model.
 * The goal is to estimate the relationship between the independent variables (x_data) and the 
 * dependent variable (y_data).
 *
 * In MLR, the model is represented by the equation:
 *   y = β0 + β1 * x1 + β2 * x2 + ... + βn * xn
 * Where:
 *   - y: Dependent variable (target)
 *   - β0: Intercept (bias)
 *   - β1, β2, ..., βn: Coefficients for the independent variables (x1, x2, ..., xn)
 *
 * Ridge regularization (also called L2 regularization) helps prevent overfitting and singularity 
 * in the inversion process by adding a penalty term to the least squares cost function:
 *   β = (X^T * X + λ * I)^(-1) * X^T * y
 * Where λ is the regularization parameter.
 *
 * Arguments:
 *    - x_data: A `gsl_matrix` representing the independent variables (X) where the matrix is of 
 *      size (m x n), with m being the number of data points and n being the number of features.
 *    - y_data: A `gsl_vector` representing the dependent variable (y) with m data points.
 *    - coeff_set: A `gsl_vector` that will store the resulting coefficients after training, 
 *      including the intercept (β0) and coefficients (β1, β2, ..., βn).
 *    - regularisation_param: A `double` representing the regularization parameter (λ) used for ridge 
 *      regularization to prevent matrix singularity and overfitting.
 *
 * Returns:
 *    - `0` if the training was successful.
 *    - `1` if there was a system error (e.g., memory allocation failure).
 *    - `2` if an arithmetic error occurred (e.g., failure during Cholesky decomposition).
*/

/*
 * Function: refine_mlr_gradient_descent
 * ----------------------------
 */
unsigned int refine_mlr_gradient_descent(gsl_matrix *x_data, gsl_vector *y_data, gsl_vector *coeff_set, unsigned int iterations, double learning_rate);
/*
 * Performs **Gradient Descent** to optimize the coefficients for a **Multiple Linear Regression (MLR)**
 * model using the standard least squares objective function. The function iteratively updates the coefficients
 * by calculating the gradient of the error with respect to the coefficients, and adjusting the coefficients
 * based on the learning rate.
 *
 * The model is represented by the equation:
 *   y = β0 + β1 * x1 + β2 * x2 + ... + βn * xn
 * Where:
 *   - y: Dependent variable (target).
 *   - β0: Intercept (bias).
 *   - β1, β2, ..., βn: Coefficients for the independent variables (x1, x2, ..., xn).
 *
 * The objective function being minimized is the **Mean Squared Error (MSE)**:
 *   MSE = (1/n) * Σ(y_i - (β0 + β1 * x1_i + ... + βn * xn_i))^2
 * Where n is the number of data points and x1_i, x2_i, ..., xn_i are the features of the i-th data point.
 *
 * The **Gradient Descent** algorithm updates the coefficients by moving in the direction opposite to the gradient
 * of the cost function:
 *   β_new = β_old - α * gradient
 * Where:
 *   - α is the learning rate (step size).
 *   - gradient is the derivative of the MSE with respect to the coefficients.
 *
 * Arguments:
 *    - x_data: A pointer to a `gsl_matrix` representing the independent variables (X), where the matrix has 
 *      dimensions (m x n), with m being the number of data points and n being the number of features.
 *    - y_data: A pointer to a `gsl_vector` representing the dependent variable (y), with m data points corresponding 
 *      to the rows of `x_data`.
 *    - coeff_set: A pointer to a `gsl_vector` that will store the resulting coefficients (including intercept) after 
 *      training. It will be updated iteratively during the gradient descent process.
 *    - iterations: An unsigned integer representing the number of iterations (or steps) to run the gradient descent 
 *      optimization process. More iterations may lead to a better approximation of the optimal coefficients.
 *    - learning_rate: A double representing the step size for the gradient descent updates. A higher learning rate may 
 *      cause the algorithm to overshoot the optimal values, while a lower learning rate may result in slower convergence.
 *
 * Returns:
 *    - `0` if the gradient descent optimization was successful and the coefficients were updated.
 *    - `1` if there was a system error (e.g., memory allocation failure).
 */


/*
 * Function: mlr_predict
 * ----------------------------
 */
void mlr_predict(gsl_matrix *x_new, gsl_vector *y_new, gsl_vector *coeff_set);
 /* This function makes predictions for new data points using the trained Multiple Linear 
 * Regression model. The model is represented by the coefficients stored in `coeff_set`, 
 * and the function computes the predicted dependent variable (y) values for new independent 
 * variables (x_new).
 *
 * The prediction is computed using the regression equation:
 *   y_new = β0 + β1 * x1_new + β2 * x2_new + ... + βn * xn_new
 *
 * Arguments:
 *    - x_new: A `gsl_matrix` representing new data points (with the same number of features 
 *      as the original training data). The matrix is of size (m_new x n), where m_new is 
 *      the number of new data points, and n is the number of independent variables.
 *    - y_new: A `gsl_vector` that will store the predicted values for the dependent variable (y_new).
 *    - coeff_set: A `gsl_vector` containing the coefficients (including intercept) from the 
 *      trained model.
 *
 * Returns:
 *    - This function does not return a value. It directly updates the `y_new` vector with the 
 *      predicted values.
 */

#endif // MLR_H
