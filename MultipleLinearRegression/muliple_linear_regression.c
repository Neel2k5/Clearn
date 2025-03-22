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

// Header link
#include "multiple_linear_regression.h"


/*
x_data -> X
y_data -> Y
coeff_set -> beta
 */
unsigned int mlr_train(gsl_matrix *x_data,gsl_vector *y_data, gsl_vector *coeff_set, double regularisation_param){

    //Allocating required local scoped vectors and matrices
    gsl_matrix *x_trans_x = gsl_matrix_alloc(x_data->size2, x_data->size2);
    gsl_vector *x_trans_y = gsl_vector_alloc(x_data->size2);  // X^T * Y
    
    if(!x_trans_x||!x_trans_y) return 1; // 1 is for system error
    // x_trans_x=(x_data^t)(x_data)
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, x_data, x_data, 0.0, x_trans_x);
    // x_trans_y=(x_data^t)(y_data)
    gsl_blas_dgemv(CblasTrans, 1.0, x_data, y_data, 0.0, x_trans_y);
   
    // doing ridge regularisation to eliminate chances of singularity for which invertion and decomposition may fail 
    for (size_t i = 0; i < x_trans_x->size1; i++) {
        gsl_matrix_set(x_trans_x, i, i, gsl_matrix_get(x_trans_x, i, i) + regularisation_param);
    }

    int decompose_status = gsl_linalg_cholesky_decomp(x_trans_x);  // Cholesky decomposition of X^T * X
    
    if(decompose_status!=GSL_SUCCESS)return 2; // 2 is for arithmatic error

    gsl_linalg_cholesky_invert(x_trans_x);  // Invert x_trans_x

    //Update coeff set by solving (x_trans_x)^-1 * (x_trans_y)
    gsl_blas_dgemv(CblasNoTrans, 1.0, x_trans_x, x_trans_y, 0.0, coeff_set); 

    // Free allocated memory
    gsl_vector_free(x_trans_y);
    gsl_matrix_free(x_trans_x);
    return 0; //No errors
}

unsigned int refine_mlr_gradient_descent(gsl_matrix*x_data,gsl_vector *y_data,gsl_vector *coeff_set, unsigned int iterations,double learning_rate){
    size_t n = x_data->size2; // number of features
    size_t m = x_data->size1; // number of data

    //Allocating required local scoped vectors 
    gsl_vector *gradient = gsl_vector_alloc(n);
    gsl_vector *prediction_set = gsl_vector_alloc(m);
    gsl_vector *error_set = gsl_vector_alloc(m);

    if(!gradient||!prediction_set||!error_set) return 1; // 1 is for system error

    for(int i=0;i<iterations;i++){
        //populate prediction_set = x_data*coeff_set
        gsl_blas_dgemv(CblasNoTrans,1,x_data,coeff_set,0,prediction_set);

        // Calculate error_set = prediction_set - y_data
        gsl_vector_memcpy(error_set,prediction_set);
        gsl_vector_sub(error_set,y_data);

        //gradient = 1/(number of data) * x_dataTranspose*error_set
        // In actual formula its 2/m but here 1 is taken for scaling
        gsl_blas_dgemv(CblasTrans,1.0/m,x_data,error_set,0,gradient);

        // coeff_set = coeff_set - learning_rate*gradient
        gsl_blas_daxpy(-learning_rate,gradient,coeff_set);

    }

    // Free allocated memory
    gsl_vector_free(gradient);
    gsl_vector_free(prediction_set);
    gsl_vector_free(error_set);

    return 0; //No errors

}

void mlr_predict(gsl_matrix *x_new,gsl_vector *y_new,gsl_vector *coeff_set){
    gsl_blas_dgemv(CblasNoTrans,1.0,x_new,coeff_set,1.0,y_new);
}
