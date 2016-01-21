CC = clang++ -I../../include
DEBUG = -g
CFLAGS = -std=c++11 -O3 -fPIC -Wall -Wextra -Werror -Weffc++ -Wstrict-aliasing --pedantic -c $(DEBUG)
MKDIR_P = mkdir -p

../../build/gmhmm.o : gmhmm.cpp
	$(MKDIR_P) ../../build/
	$(CC) $(CFLAGS) gmhmm.cpp -o ../../build/gmhmm.o

../../build/phi.o : phi.cpp
	$(MKDIR_P) ../../build/
	$(CC) $(CFLAGS) phi.cpp -o ../../build/phi.o

.PHONY : clean
clean :
	-rm -f ../../build/*.o *~ hmm &> /dev/null
