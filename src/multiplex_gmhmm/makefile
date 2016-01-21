CC = clang++ -I../../include
DEBUG = -g
CFLAGS = -std=c++11 -O3 -fPIC -Wall -Wextra -Werror -Weffc++ -Wstrict-aliasing --pedantic -c $(DEBUG)
MKDIR_P = mkdir -p

../../build/multiplex_gmhmm.o : multiplex_gmhmm.cpp
	$(MKDIR_P) ../../build/
	$(CC) $(CFLAGS) multiplex_gmhmm.cpp -o ../../build/multiplex_gmhmm.o

.PHONY : clean 
clean :
	-rm -f ../../build/*.o *~ multiplex &> /dev/null
