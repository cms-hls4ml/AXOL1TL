.PHONY: clean

GTADModel_v6.so: emulator.o GTADModel_v6_project.o
	$(CXX) $(CXXFLAGS) $(LD_FLAGS) -shared $^ -o $@

%.o: NN/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f emulator.o GTADModel_v6_project.o GTADModel_v6.so