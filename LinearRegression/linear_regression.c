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


#include "linear_regression.h"

void linear_regression(LinearRegressionResult *result_set, double *x_data, double *y_data, int number_of_data){
    // To calculate the slope the formula is slope=(mean(xy)-mean(x)mean(y))/(mean(x^2)-mean(x)^2)
    double slope = 0,
           x_mean = 0,
           y_mean = 0,
           xy_mean = 0,
           x_squared_mean = 0,
           intercept = 0;

    for(int i=0;i<number_of_data;i++){
        x_mean+=x_data[i];
        y_mean+=y_data[i];
        xy_mean+=(x_data[i]*y_data[i]);
        x_squared_mean+=(x_data[i]*x_data[i]);
    }
    x_mean /= number_of_data;
    y_mean /= number_of_data;
    xy_mean /= number_of_data;
    x_squared_mean /= number_of_data;

    slope=(xy_mean-(x_mean*y_mean))/(x_squared_mean-(x_mean*x_mean));

    // To calculate the intercept k=mean(y)-(slope*mean(x))
    intercept=y_mean-(slope*x_mean);

    result_set->slope=slope;
    result_set->intercept=intercept;
    //result might be slightly different from manually calculated result due to floating point precision 
   
}
