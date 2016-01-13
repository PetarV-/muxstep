CC = clang++ -I../../include
DEBUG = -g
CFLAGS = -std=c++11 -O3 -Wall -Wextra -Werror -Weffc++ -Wstrict-aliasing --pedantic -c $(DEBUG)
MKDIR_P = mkdir -p

vector_cmp : vector_cmp.cpp
	$(MKDIR_P) ../../build/
	$(CC) $(CFLAGS) vector_cmp.cpp -o ../../build/vector_cmp.o

clean :
	-rm -f ../../build/*.o *~ vector_cmp &> /dev/null
