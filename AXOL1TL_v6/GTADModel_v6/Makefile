.PHONY: clean

MODEL_NAME = GTADModel_v6

$(MODEL_NAME).so: $(MODEL_NAME)_project.o emulator.o
	$(CXX) $(CXXFLAGS) $(LD_FLAGS) -shared $^ -o $@

%.o: NN/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(MODEL_NAME)_project.o emulator.o $(MODEL_NAME).so
