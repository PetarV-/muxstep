OBJS = ../../build/syn_gen.o
CC = clang++ -I../../include
DEBUG = -g
CFLAGS = -std=c++11 -O3 -fPIC -Wall -Wextra -Werror -Weffc++ -Wstrict-aliasing --pedantic -c $(DEBUG)
MKDIR_P = mkdir -p

../../bin/syn_gen : $(OBJS)
	$(MKDIR_P) ../../bin/
	$(CC) $(LFLAGS) $(OBJS) -o ../../bin/syn_gen

../../build/syn_gen.o : syn_gen.cpp
	$(MKDIR_P) ../../build/
	$(CC) $(CFLAGS) syn_gen.cpp -o ../../build/syn_gen.o

.PHONY : clean
clean :
	-rm -f ../../build/*.o *~ ../../bin/syn_gen &> /dev/null

