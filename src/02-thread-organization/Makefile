TEST_SOURCE = xxx.cu

TARGETBIN := ./xxx

cc = nvcc

$(TARGETBIN) := $(TEST_SOURCE)
	$(cc) $(TEST_SOURCE) -o $(TARGETBIN)

.PHONY:clean

clean: 
	rm -rf $(TARGETBIN)
	rn -rf *.o