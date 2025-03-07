.PHONY: clean

GTADModel_v5.so: GTADModel_emulator_v5.o GTADModel_v5.o
	$(CXX) $(CXXFLAGS) $(LD_FLAGS) -shared $^ -o $@

%.o: NN/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm GTADModel_emulator_v5.o GTADModel_v5.o GTADModel_v5.so
