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


#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H


// GSL headers necessary for linear algebra with scope to this model
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
// math.h for exponential
#include <math.h>


/*
 * Function: sigmoid
 */

double sigmoid(double x);
/* ----------------------------
 * This function implements the sigmoid activation function, which is commonly 
 * used in logistic regression models to map input values to a probability between 0 and 1.
 * 
 * The function applies the sigmoid function on the input `x` as follows:
 *    sigmoid(x) = 1 / (1 + exp(-x))
 * 
 * The sigmoid function is used to transform linear predictions (the weighted sum of inputs)
 * into a probability value between 0 and 1.
 * 
 * Arguments:
 *    - x: A double representing the input value to the sigmoid function.
 *
 * Returns:
 *    - A double, which is the result of applying the sigmoid function to the input `x`.
 *
 * Example:
 *    double result = sigmoid(0.5); // result will be approximately 0.622
 */


/*
 * Function: logistic_regression_train
 */

unsigned int logistic_regression_train(gsl_vector *y_data, gsl_matrix *x_data, gsl_vector *weight_set, double learning_rate, unsigned int iterations);
/* ----------------------------
 * This function trains a logistic regression model using the gradient descent 
 * optimization method. The model predicts the probability of a binary outcome 
 * based on a set of features (x_data). The function updates the weight_set vector 
 * (which holds the model's weights) by iterating through the training data 
 * and minimizing the logistic loss function using gradient descent.
 *
 * The model uses the sigmoid function to predict the probability of the positive class 
 * for each data point. The gradient descent algorithm is used to update the weights 
 * in the direction that minimizes the loss.
 * 
 * The function performs the following steps:
 * 1. Calculate the linear combination (z_data = x_data * weight_set)
 * 2. Apply the sigmoid function to get predicted probabilities (y_predicted)
 * 3. Compute the error (y_predicted - y_data)
 * 4. Compute the gradient as the dot product of the transposed x_data and the error
 * 5. Update the weights (weight_set) by subtracting the scaled gradient.
 *
 * Arguments:
 *    - y_data: A pointer to a `gsl_vector` containing the true labels (target variable).
 *    - x_data: A pointer to a `gsl_matrix` containing the feature set (input variables).
 *    - weight_set: A pointer to a `gsl_vector` containing the model's weights, which will 
 *      be updated during the training process.
 *    - learning_rate: A double representing the learning rate used in gradient descent.
 *    - iterations: The number of iterations to perform the gradient descent optimization.
 *
 * Returns:
 *    - An unsigned integer (0 if training completes successfully, 1 if memory allocation fails).
 *
 * Notes:
 *    - This function uses GSL (GNU Scientific Library) for matrix and vector operations.
 *    - The function modifies the `weight_set` vector in-place during training.
 *    - It assumes that `y_data`, `x_data`, and `weight_set` are valid and properly allocated.
 *    - If memory allocation fails for vectors used in intermediate computations, the function returns 1.
 *    - The learning rate and number of iterations should be chosen carefully for optimal training.
 */


/*
 * Function: probability
 */

int probability(double x);
/* ----------------------------
 * This function converts a continuous probability value into a binary classification.
 * It uses a threshold of 0.5 to decide the predicted class:
 *    - If x > 0.5, the predicted class is 1 (positive class).
 *    - If x <= 0.5, the predicted class is 0 (negative class).
 *
 * This function is commonly used in logistic regression models to convert predicted 
 * probabilities into class labels.
 *
 * Arguments:
 *    - x: A double representing the probability value to be converted into a class label.
 *
 * Returns:
 *    - An integer (0 or 1), representing the predicted class.
 *
 * Example:
 *    int predicted_class = probability(0.75); // predicted_class will be 1
 *    int predicted_class = probability(0.25); // predicted_class will be 0
 */


/*
 * Function: logistic_regression_predict
 */

unsigned int logistic_regression_predict(gsl_vector *prediction_set, gsl_vector *weight_set, gsl_matrix *x_data);
/* ----------------------------
 * This function makes predictions using a logistic regression model. The function calculates 
 * the probability of the positive class for each data point in `x_data` based on the trained 
 * weights stored in `weight_set`. The predictions are then converted to binary class labels 
 * using the `probability` function.
 *
 * The function performs the following steps:
 * 1. Compute the linear combination (z_data = x_data * weight_set)
 * 2. Apply the sigmoid function to compute the predicted probability for each data point.
 * 3. Convert the predicted probability into a binary class label (0 or 1) using the `probability` function.
 *
 * Arguments:
 *    - prediction_set: A pointer to a `gsl_vector` where the predicted class labels will be stored.
 *    - weight_set: A pointer to a `gsl_vector` containing the trained model's weights.
 *    - x_data: A pointer to a `gsl_matrix` containing the input feature set for which predictions 
 *      are to be made.
 *
 * Returns:
 *    - An unsigned integer (0 if prediction completes successfully, 1 if memory allocation fails).
 *
 * Notes:
 *    - This function uses the trained weights from `weight_set` to make predictions.
 *    - The function assumes that `x_data`, `weight_set`, and `prediction_set` are properly allocated.
 *    - If memory allocation fails for the intermediate `z_data` vector, the function returns 1.
 *    - The function applies the logistic regression model to each data point in `x_data` and 
 *      stores the predicted class labels in `prediction_set`.
 */


#endif // LOGISTIC_REGRESSION_H
