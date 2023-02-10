#!/bin/bash

CC=g++
CFLAGS="-O3 -fPIC -std=c++17"

LDFLAGS=
INCFLAGS="-I../ap_types/"
PROJECT=anomaly_detection

${CC} ${CFLAGS} ${INCFLAGS} -c firmware/${PROJECT}.cpp -o ${PROJECT}.o
${CC} ${CFLAGS} ${INCFLAGS} -c ${PROJECT}_emulator.cpp -o ${PROJECT}_emulator.o
${CC} ${CFLAGS} ${INCFLAGS} -shared ${PROJECT}.o ${PROJECT}_emulator.o -o ../${PROJECT}.so
rm -f *.o
