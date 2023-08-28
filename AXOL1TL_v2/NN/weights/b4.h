//Numpy array shape [16]
//Min -1.625000000000
//Max 4.125000000000
//Number of zeros 2

#ifndef B4_H_
#define B4_H_

#ifdef LOAD_WEIGHTS_FROM_TXT
bias4_t b4[16];
#else
bias4_t b4[16] = {1.250, 0.500, 0.250, 1.125, 0.625, -0.625, -0.125, -0.500, 0.125, 0.250, 0.375, 0.250, -0.250, 0.625, 0.500, 0.500};
#endif

#endif
