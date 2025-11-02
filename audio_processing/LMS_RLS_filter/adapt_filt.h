/***************************************************************************
  *
  * Homework for chapter 4 -- Adaptive filter using LMS & RLS
  *
  * Here is the declaration of adapt_filtering function.
  *
  **************************************************************************/

#ifndef _ADAPT_FILT_H_
#define _ADAPT_FILT_H_

// Algorithm selection: 0 for LMS, 1 for RLS
#define USE_RLS 0

const int filter_length = 128;
static double inputdata[filter_length];

#if USE_RLS
// RLS algorithm variables
static double P[filter_length][filter_length];  // Inverse correlation matrix
static int rls_initialized = 0;
#endif

int adapt_filtering(short input, double* adapt_filter, int filter_length, short* err);

#endif

