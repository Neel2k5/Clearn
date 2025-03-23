

// GSL headers necessary for linear algebra with scope to this model
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
// math.h for exponential
#include <math.h>

#include "logistic_regression.h"

int main() {
    // Example data (3 samples, 2 features)
    gsl_matrix *x_data = gsl_matrix_alloc(3, 2); // 3 samples, 2 features
    gsl_vector *y_data = gsl_vector_alloc(3);    // 3 labels (0 or 1)
    gsl_vector *weight_set = gsl_vector_alloc(2); // 2 weights (1 for each feature)
    gsl_vector *prediction_set = gsl_vector_alloc(3); // To store predictions

    // Set up example data
    gsl_matrix_set(x_data, 0, 0, 1.0); gsl_matrix_set(x_data, 0, 1, 2.0);
    gsl_matrix_set(x_data, 1, 0, 1.0); gsl_matrix_set(x_data, 1, 1, 3.0);
    gsl_matrix_set(x_data, 2, 0, 2.0); gsl_matrix_set(x_data, 2, 1, 1.0);

    gsl_vector_set(y_data, 0, 0.0);  // Label 0
    gsl_vector_set(y_data, 1, 1.0);  // Label 1
    gsl_vector_set(y_data, 2, 0.0);  // Label 0

    // Initialize weights (zero initialization)
    gsl_vector_set(weight_set, 0, 0.0);
    gsl_vector_set(weight_set, 1, 0.0);

    // Train the model
    double learning_rate = 0.001;
    unsigned int iterations = 10000;
    unsigned int result = logistic_regression_train(y_data, x_data, weight_set, learning_rate, iterations);

    // Check if the training was successful
    if (result == 0) {
        printf("Training successful!\n");
        printf("Final weights: %f, %f\n", gsl_vector_get(weight_set, 0), gsl_vector_get(weight_set, 1));
    } else {
        printf("Training failed!\n");
    }

    // Make predictions using the trained model
    result = logistic_regression_predict(prediction_set, weight_set, x_data);

    if (result == 0) {
        // Output predictions and compare with true labels
        printf("\nPredictions:\n");
        for (int i = 0; i < prediction_set->size; i++) {
            printf("Sample %d: Predicted label: %f, True label: %f\n", i, gsl_vector_get(prediction_set, i), gsl_vector_get(y_data, i));
        }
    } else {
        printf("Prediction failed!\n");
    }

    // Free allocated memory
    gsl_matrix_free(x_data);
    gsl_vector_free(y_data);
    gsl_vector_free(weight_set);
    gsl_vector_free(prediction_set);
    return 0;
}
