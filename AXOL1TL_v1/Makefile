.PHONY: clean

GTADModel.so: GTADModel_emulator.o GTADModel.o
	$(CXX) $(CXXFLAGS) $(LD_FLAGS) -shared $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm GTADModel.so
