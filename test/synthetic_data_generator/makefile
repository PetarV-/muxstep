OBJS = ../../build/syn_gen.o
CC = clang++ -I../../include
DEBUG = -g
CFLAGS = -std=c++11 -O3 -Wall -Wextra -Werror -Weffc++ -Wstrict-aliasing --pedantic -c $(DEBUG)
MKDIR_P = mkdir -p

syn_gen : $(OBJS)
	$(MKDIR_P) ../../bin/
	$(CC) $(LFLAGS) $(OBJS) -o ../../bin/syn_gen

../../build/syn_gen.o : syn_gen.cpp
	$(MKDIR_P) ../../build/
	$(CC) $(CFLAGS) syn_gen.cpp -o ../../build/syn_gen.o

clean :
	-rm -f ../../build/*.o *~ ../../bin/syn_gen &> /dev/null

