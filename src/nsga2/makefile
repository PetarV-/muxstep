CC = clang++ -I../../include
DEBUG = -g
CFLAGS = -std=c++11 -O3 -Wall -Wextra -Werror -Weffc++ -Wstrict-aliasing --pedantic -c $(DEBUG)
MKDIR_P = mkdir -p

nsga2 : nsga2.cpp
	$(MKDIR_P) ../../build/
	$(CC) $(CFLAGS) nsga2.cpp -o ../../build/nsga2.o

clean :
	-rm -f ../../build/*.o *~ nsga2 &> /dev/null
