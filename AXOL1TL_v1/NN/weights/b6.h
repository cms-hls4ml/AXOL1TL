//Numpy array shape [13]
//Min -0.750000000000
//Max 2.125000000000
//Number of zeros 4

#ifndef B6_H_
#define B6_H_

#ifdef LOAD_WEIGHTS_FROM_TXT
bias6_t b6[13];
#else
bias6_t b6[13] = {0.250, 1.625, -0.125, 0.000, 0.250, 0.375, 0.000, 0.000, 0.500, 2.125, 0.125, -0.750, 0.000};
#endif

#endif
