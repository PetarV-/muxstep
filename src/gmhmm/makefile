CC = clang++ -I../../include
DEBUG = -g
CFLAGS = -std=c++11 -O3 -Wall -Wextra -Werror -Weffc++ -Wstrict-aliasing --pedantic -c $(DEBUG)
MKDIR_P = mkdir -p

gmhmm : gmhmm.cpp
	$(MKDIR_P) ../../build/
	$(CC) $(CFLAGS) gmhmm.cpp -o ../../build/gmhmm.o

phi : phi.cpp
	$(MKDIR_P) ../../build/
	$(CC) $(CFLAGS) phi.cpp -o ../../build/phi.o

clean :
	-rm -f ../../build/*.o *~ hmm &> /dev/null
