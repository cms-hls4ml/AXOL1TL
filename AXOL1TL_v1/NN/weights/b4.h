//Numpy array shape [16]
//Min -1.625000000000
//Max 4.125000000000
//Number of zeros 2

#ifndef B4_H_
#define B4_H_

#ifdef LOAD_WEIGHTS_FROM_TXT
bias4_t b4[16];
#else
bias4_t b4[16] = {0.250, 0.000, 0.125, 4.125, 0.625, 2.500, 1.125, 0.250, 0.375, 0.375, 0.000, -0.500, 2.750, 1.625, 0.250, -1.625};
#endif

#endif
