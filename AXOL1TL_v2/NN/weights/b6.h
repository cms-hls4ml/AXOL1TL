//Numpy array shape [13]
//Min -0.750000000000
//Max 2.125000000000
//Number of zeros 4

#ifndef B6_H_
#define B6_H_

#ifdef LOAD_WEIGHTS_FROM_TXT
bias6_t b6[8];
#else
bias6_t b6[8] = {-0.125, 0.000, 0.250, -0.625, 0.375, 0.000, 0.500, 0.625};
#endif

#endif
