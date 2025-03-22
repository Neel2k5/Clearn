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

#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include "multiple_linear_regression.h"  // Include the MLR header for function declarations

int main() {
    int num_data_points = 5;   // Number of data points
    int num_features = 4;      // Number of features (including the intercept)

    // Allocate memory for the dataset (X and Y)
    gsl_matrix *x_data = gsl_matrix_alloc(num_data_points, num_features);
    gsl_vector *y_data = gsl_vector_alloc(num_data_points);
    gsl_vector *coeff_set = gsl_vector_alloc(num_features);  // For storing coefficients (beta)

    // Example Data (5 data points and 4 features)
    double x_data_array[5][4] = {
        {1, 2, 5, 7},
        {1, 3, 6, 8},
        {1, 4, 7, 9},
        {1, 5, 8, 10},
        {1, 6, 9, 11}
    };

    double y_data_array[5] = {50, 60, 70, 80, 90}; // Target values (Y)

    // Populate datasets
    for (int i = 0; i < num_data_points; i++) {
        for (int j = 0; j < num_features; j++) {
            gsl_matrix_set(x_data, i, j, x_data_array[i][j]);
        }
    }
    for (int i = 0; i < num_data_points; i++) {
        gsl_vector_set(y_data, i, y_data_array[i]);
    }

    // Set regularization parameter for Ridge regression 
    double regularisation_param = 0.1; // Regularization parameter

    // First, train the model using mlr_train
    unsigned int result = mlr_train(x_data, y_data, coeff_set, regularisation_param);

    // Check if mlr_train was successful
    if (result == 0) {
        printf("Training successful using mlr_train! Initial Coefficients:\n");
        for (int i = 0; i < coeff_set->size; i++) {
            printf("Coefficient %d: %.4f\n", i, gsl_vector_get(coeff_set, i));
        }
    } else {
        printf("Training failed with error code: %u\n", result);
        gsl_matrix_free(x_data);
        gsl_vector_free(y_data);
        gsl_vector_free(coeff_set);
        return 1;  // Exit if training failed
    }

    // Now, refine the coefficients using refine_mlr_gradient_descent
    double learning_rate = 0.01;   // Learning rate for gradient descent
    unsigned int iterations = 1000; // Number of iterations for gradient descent

    result = refine_mlr_gradient_descent(x_data, y_data, coeff_set, iterations, learning_rate);

    // Check if refine_mlr_gradient_descent was successful
    if (result == 0) {
        printf("\nRefining coefficients using Gradient Descent completed! Refined Coefficients:\n");
        for (int i = 0; i < coeff_set->size; i++) {
            printf("Coefficient %d: %.4f\n", i, gsl_vector_get(coeff_set, i));
        }
    } else {
        printf("Refining failed with error code: %u\n", result);
    }

    // Test prediction on new data
    int num_test_points = 2;  // Let's predict 2 new data points

    gsl_matrix *x_new = gsl_matrix_alloc(num_test_points, num_features);  // New data matrix
    gsl_vector *y_new = gsl_vector_alloc(num_test_points);  // Predicted values

    // Example test data (2 new data points)
    double x_new_array[2][4] = {
        {1, 7, 10, 13}, 
        {1, 8, 11, 14}   
    };

    // Populate the new data matrix
    for (int i = 0; i < num_test_points; i++) {
        for (int j = 0; j < num_features; j++) {
            gsl_matrix_set(x_new, i, j, x_new_array[i][j]);
        }
    }

    // Use the refined model to make predictions on the new data
    mlr_predict(x_new, y_new, coeff_set);

    printf("\nPredictions for new data points:\n");
    for (int i = 0; i < y_new->size; i++) {
        printf("y_new(%d) = %.4f\n", i, gsl_vector_get(y_new, i));
    }

    // Free allocated memory
    gsl_matrix_free(x_data);
    gsl_vector_free(y_data);
    gsl_vector_free(coeff_set);
    gsl_matrix_free(x_new);
    gsl_vector_free(y_new);

    return 0;
}
