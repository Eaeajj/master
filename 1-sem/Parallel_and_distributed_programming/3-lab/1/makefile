CC=nvcc
CFLAGS=-I.
DEPS = calcs.h helpers.h cuda_helpers.h
OBJ = main.o calcs.o cuda_helpers.o helpers.o


%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

run: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)
	@make clean-objects

clean-objects:
	rm -f ./*.o

clean:
	@make clean-objects
	rm -f run
