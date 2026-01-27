.PHONY: clean

GTADModel_v6.so: GTADModel_emulator_v6.o GTADModel_v6.o
	$(CXX) $(CXXFLAGS) $(LD_FLAGS) -shared $^ -o $@

%.o: NN/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm GTADModel_emulator_v6.o GTADModel_v6.o GTADModel_v6.so