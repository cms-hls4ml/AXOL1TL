.PHONY: clean

GTADModel_v1.so: ./GTADModel_hls/GTADModel.o GTADModel_v1.o
	$(CXX) $(CXXFLAGS) $(LD_FLAGS) -shared $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm GTADModel_v1.so
