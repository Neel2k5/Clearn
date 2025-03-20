#include <stdio.h>
#include <stdlib.h>

#include "linear_regression.h"

int main(){
    //50 points
    double x_data[] = {2.35, 3.46, 1.23, 4.57, 5.68, 6.79, 7.89, 8.90, 9.01, 10.12,
                   11.23, 12.34, 13.46, 14.57, 15.68, 16.79, 17.89, 18.90, 19.01, 20.12,
                   21.23, 22.34, 23.46, 24.57, 25.68, 26.79, 27.89, 28.90, 29.01, 30.12,
                   31.23, 32.34, 33.46, 34.57, 35.68, 36.79, 37.89, 38.90, 39.01, 40.12,
                   41.23, 42.34, 43.46, 44.57, 45.68, 46.79, 47.89, 48.90, 49.01, 50.12};

    double y_data[] = {4.23, 5.35, 3.23, 6.57, 7.68, 8.79, 9.89, 10.90, 11.01, 12.12,
                   13.23, 14.35, 15.46, 16.57, 17.68, 18.79, 19.89, 20.90, 21.01, 22.12,
                   23.23, 24.35, 25.46, 26.57, 27.68, 28.79, 29.89, 30.90, 31.01, 32.12,
                   33.23, 34.35, 35.46, 36.57, 37.68, 38.79, 39.89, 40.90, 41.01, 42.12,
                   43.23, 44.35, 45.46, 46.57, 47.68, 48.79, 49.89, 50.90, 51.01, 52.12};

    LinearRegressionResult result;
    linear_regression(&result,x_data,y_data,50);
    printf("\nResult:\nslope = %lf\nintercept = %lf\n",result.slope,result.intercept);
    double new_x=7.908;
    double predicted_y = result.slope*new_x + result.intercept;
    printf("\nPredicted y for x = %lf is y=%lf\n",new_x,predicted_y);

    

    return 0;
}