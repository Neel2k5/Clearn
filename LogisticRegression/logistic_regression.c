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

// GSL headers necessary for linear algebra with scope to this model
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
// math.h for exponential
#include <math.h>

#include "logistic_regression.h"

double sigmoid(double x){
    return 1/(1+exp(-x));
}

unsigned int logistic_regression_train(gsl_vector *y_data, gsl_matrix *x_data, gsl_vector *weight_set, double learning_rate, unsigned int iterations) {
    int m = y_data->size; // number of data points (rows)
    int n = x_data->size2; // number of features (columns)
    gsl_vector *z_data = gsl_vector_alloc(m);
    gsl_vector *gradient = gsl_vector_alloc(n); 
    gsl_vector *y_predicted = gsl_vector_alloc(m);
    gsl_vector *error_set = gsl_vector_alloc(m);
    if (!z_data || !gradient || !y_predicted||!error_set) {
        if (z_data) gsl_vector_free(z_data);
        if (gradient) gsl_vector_free(gradient);
        if (y_predicted) gsl_vector_free(y_predicted);
        if (error_set) gsl_vector_free(error_set);
        return 1;
    }

    for (unsigned int i = 0; i < iterations; i++) {
        // z_data = x_data * weight_set
        gsl_blas_dgemv(CblasNoTrans, 1, x_data, weight_set, 0, z_data);

        // Predict values with current weights
        for (unsigned int j = 0; j < m; j++) {
            double sig = sigmoid(gsl_vector_get(z_data, j));
            gsl_vector_set(y_predicted, j, sig);
        }
        // Compute gradient = x_data_transpose * (y_predicted - y_data)
        // Create a vector that holds the error_set (y_predicted - y_data)
        
        gsl_vector_memcpy(error_set, y_predicted);
        gsl_vector_sub(error_set, y_data);

        // Compute gradient = x_data_transpose * error_set
        gsl_blas_dgemv(CblasTrans, 1, x_data, error_set, 0, gradient); // gradient = X^T * (y_pred - y)

        // Update weight_set = weight_set - (learning_rate / m) * gradient
        gsl_vector_scale(gradient, learning_rate / m);
        gsl_vector_sub(weight_set, gradient);

        
    }
    gsl_vector_free(error_set);
    gsl_vector_free(z_data);
    gsl_vector_free(gradient);
    gsl_vector_free(y_predicted);
    return 0;
}

int probability(double x){
    return (x>0.5)?1:0;
}
unsigned int logistic_regression_predict(gsl_vector *prediction_set,gsl_vector *weight_set, gsl_matrix *x_data){
    int n = x_data->size1;
    gsl_vector *z_data = gsl_vector_alloc(n);
    if(!z_data)return 1;
    // z_data = x_data * weight_set
    gsl_blas_dgemv(CblasNoTrans, 1, x_data, weight_set, 0, z_data);

    for(int i=0;i<n;i++){
        double sig = sigmoid(gsl_vector_get(z_data,i));
        gsl_vector_set(prediction_set,i,probability(sig));
    }
    return 0;
    
}
