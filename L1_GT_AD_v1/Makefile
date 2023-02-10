INCFLAGS="-Iap_types/"

example_emulator: main.cpp emulator.h
	$(CXX) $(CXXFLAGS) $(INCFLAGS) -o example_emulator -std=c++17 main.cpp -ldl

clean:
	rm -f example_emulator

.PHONY: clean
