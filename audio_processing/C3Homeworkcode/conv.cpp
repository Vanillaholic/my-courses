/***************************************************************************
  * 
  * Homework for chapter 3 -- "overlap-save method"
  *
  * Here is the realization of add rir function.
  * You have to finish "conv" function by yourself. It is called in main 
  * function. Also, you may want to use FFT, which is ready for you too.
  *
  * NOTICE: There is no IFFT function. So you have to realize IFFT using FFT.
  * Take some time to think about how to do this please.
  * 
  **************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "conv.h"
#include "baselib.h"

/**
 * @brief add rir using overlap-save method.
 *
 * @param inputdata         input clean audio data buffer
 * @param inputdata_length  length of inputdata(samples)
 * @param rir               room impulse response buffer
 * @param rir_length        length of rir, 4096 by default
 * @param outputdata        processed data, same length as inputdata
 * @return 
 *     @retval 0            successfully
 */
int conv(short* inputdata, long inputdata_length, double* rir, long rir_length, short* outputdata)
{
    // finish overlap-save method here

    // Step 1: Determine FFT size (must be power of 2 and >= rir_length)
    int fftorder = 0;
    long fftsize = 1;
    while (fftsize < rir_length * 2) {
        fftsize *= 2;
        fftorder++;
    }

    // Block size for overlap-save (FFT size - RIR length + 1)
    long blocksize = fftsize - rir_length + 1;

    // Step 2: Prepare RIR in frequency domain
    COMPLEX* rir_fft = new COMPLEX[fftsize];
    memset(rir_fft, 0, sizeof(COMPLEX) * fftsize);

    // Copy RIR data and zero-pad
    for (long i = 0; i < rir_length; i++) {
        rir_fft[i].real = (float)rir[i];
        rir_fft[i].image = 0.0f;
    }

    // FFT of RIR
    fft(rir_fft, fftorder);

    // Step 3: Allocate buffers for processing
    COMPLEX* input_block = new COMPLEX[fftsize];
    COMPLEX* output_block = new COMPLEX[fftsize];

    // Step 4: Process input data block by block using overlap-save
    long num_blocks = (inputdata_length + blocksize - 1) / blocksize;

    for (long block = 0; block < num_blocks; block++) {
        // Clear input block buffer
        memset(input_block, 0, sizeof(COMPLEX) * fftsize);

        // Copy overlapping part from previous block (rir_length - 1 samples)
        long overlap_start = block * blocksize - (rir_length - 1);

        // Fill input block with data
        for (long i = 0; i < fftsize; i++) {
            long input_idx = overlap_start + i;
            if (input_idx < inputdata_length) {
                input_block[i].real = (float)inputdata[input_idx];
            } else {
                input_block[i].real = 0.0f;
            }
            input_block[i].image = 0.0f;
        }

        // FFT of input block
        fft(input_block, fftorder);

        // Frequency domain multiplication (convolution in time domain)
        for (long i = 0; i < fftsize; i++) {
            float real_part = input_block[i].real * rir_fft[i].real -
                            input_block[i].image * rir_fft[i].image;
            float imag_part = input_block[i].real * rir_fft[i].image +
                            input_block[i].image * rir_fft[i].real;
            output_block[i].real = real_part;
            output_block[i].image = imag_part;
        }

        // IFFT using FFT (conjugate method)
        // IFFT(X) = conj(FFT(conj(X))) / N

        // Step 1: Conjugate
        for (long i = 0; i < fftsize; i++) {
            output_block[i].image = -output_block[i].image;
        }

        // Step 2: FFT
        fft(output_block, fftorder);

        // Step 3: Conjugate and scale
        for (long i = 0; i < fftsize; i++) {
            output_block[i].image = -output_block[i].image;
            output_block[i].real /= fftsize;
            output_block[i].image /= fftsize;
        }

        // Extract valid output (discard first rir_length-1 samples)
        long valid_start = rir_length - 1;
        long valid_length = blocksize;

        for (long i = 0; i < valid_length; i++) {
            long output_idx = block * blocksize + i;
            if (output_idx < inputdata_length) {
                // Round and clip to short range
                float val = output_block[valid_start + i].real;
                if (val > 32767.0f) val = 32767.0f;
                if (val < -32768.0f) val = -32768.0f;
                outputdata[output_idx] = (short)(val + 0.5f);
            }
        }
    }

    // Step 5: Clean up
    delete[] rir_fft;
    delete[] input_block;
    delete[] output_block;

    return 0;
}

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
