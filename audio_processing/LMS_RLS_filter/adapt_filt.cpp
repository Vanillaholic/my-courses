/***************************************************************************
  *
  * Homework for chapter 4 -- Adaptive filter using LMS & RLS
  *
  * Here is the realization of adapt_filtering function.
  *
  **************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "adapt_filt.h"

/**
 * @brief time-domain adaptive filter algorithm
 *
 * @param input          input audio sample of time index(n)
 * @param adapt_filter   adaptive filter buffer
 * @param filter_length  adaptive filter length, 128 by default
 * @param err            error output of time index(n)
 * @return
 *     @retval 0         successfully
 */
int adapt_filtering(short input, double* adapt_filter, int filter_length, short* err) {
    int i, j;
    double filter_output = 0.0;
    double error = 0.0;
    double desired = 0.0;

    inputdata[0] = double(input / 32768.0);

    // 作业中期望信号等于输入信号,这样理想状态滤波器为Delta函数
    desired = inputdata[0];

    // finish adaptive filtering algorithm here, using LMS and RLS, respectively

#if USE_RLS
    // ============== RLS Algorithm ==============
    // 相关参数
    const double lambda = 0.99;  // 遗忘因子 (0.95 ~ 0.99)
    const double delta = 1.0;    // P矩阵初始化的值


    // 初始化P矩阵
    if(!rls_initialized) {
        for(i = 0; i < filter_length; i++) {
            for(j = 0; j < filter_length; j++) {
                P[i][j] = (i == j) ? (1.0 / delta) : 0.0;  // P(0) = (1/delta) * I
            }
        }
        rls_initialized = 1;
    }

    // 1. 计算滤波器输出 y(n) = w^T(n-1) * x(n)
    filter_output = 0.0;
    for(i = 0; i < filter_length; i++) {
        filter_output += adapt_filter[i] * inputdata[i];
    }

    // 2.计算误差: e(n) = d(n) - y(n)
    error = desired - filter_output;

    // 3. 计算增益向量 k(n) = P(n-1) * x(n) / (lambda + x^T(n) * P(n-1) * x(n))
    double Px[filter_length];      // P(n-1) * x(n)
    double xTPx = 0.0;             // x^T(n) * P(n-1) * x(n)

    //  Px = P(n-1) * x(n)
    for(i = 0; i < filter_length; i++) {
        Px[i] = 0.0;
        for(j = 0; j < filter_length; j++) {
            Px[i] += P[i][j] * inputdata[j];
        }
    }

    //  xTPx = x^T * Px
    for(i = 0; i < filter_length; i++) {
        xTPx += inputdata[i] * Px[i];
    }

    // 增益向量 k(n)
    double k[filter_length];
    double denominator = lambda + xTPx;
    for(i = 0; i < filter_length; i++) {
        k[i] = Px[i] / denominator;
    }

    // 4. 更新滤波器系数: w(n) = w(n-1) + k(n) * e(n)
    for(i = 0; i < filter_length; i++) {
        adapt_filter[i] += k[i] * error;
    }

    // 5. 更新P矩阵: P(n) = (1/lambda) * (P(n-1) - k(n) * x^T(n) * P(n-1))
    // 计算 kxTP = k(n) * x^T(n) * P(n-1)
    double kxTP[filter_length][filter_length];
    for(i = 0; i < filter_length; i++) {
        for(j = 0; j < filter_length; j++) {
            kxTP[i][j] = k[i] * Px[j];
        }
    }

    // 计算 P: P(n) = (1/lambda) * (P(n-1) - kxTP)
    for(i = 0; i < filter_length; i++) {
        for(j = 0; j < filter_length; j++) {
            P[i][j] = (P[i][j] - kxTP[i][j]) / lambda;
        }
    }

#else
    // ============== LMS Algorithm ==============
    // 滤波器步长
    const double mu = 0.18;

    // 1. 计算滤波器输出 y(n) = w^T(n) * x(n)
    filter_output = 0.0;
    for(i = 0; i < filter_length; i++) {
        filter_output += adapt_filter[i] * inputdata[i];
    }

    // 2.计算误差: e(n) = d(n) - y(n)
    error = desired - filter_output;

    // 3. 更新滤波器系数: w(n+1) = w(n) + mu * e(n) * x(n)
    for(i = 0; i < filter_length; i++) {
        adapt_filter[i] += mu * error * inputdata[i];
    }
#endif

    // end adaptive filtering algorithm

    // output error
    err[0] = short(error * 32768.0);

    return 0;
}
