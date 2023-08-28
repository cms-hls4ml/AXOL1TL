# ANOMALY DETECTION FOR GLOBAL TRIGGER

Note: emulator does not load weights from text files.
After generating project from hls4ml, use `#ifdef LOAD_WEIGHTS_FROM_TXT` instead of `#ifndef __SYNTHESIS__`

```bash
sed -i "s/#ifndef __SYNTHESIS__/#ifdef LOAD_WEIGHTS_FROM_TXT/g" *.h
```
