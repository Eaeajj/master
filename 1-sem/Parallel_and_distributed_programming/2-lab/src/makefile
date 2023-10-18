CC=nvcc
CFLAGS=-I.
DEPS = calcs.h dev_array.h
OBJ = main.o calcs.o


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
