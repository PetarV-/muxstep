CC = clang++ -I../../include
DEBUG = -g
CFLAGS = -std=c++11 -O3 -fPIC -Wall -Wextra -Werror -Weffc++ -Wstrict-aliasing --pedantic -c $(DEBUG)
MKDIR_P = mkdir -p

../../build/classifier_multiplex_gmhmm.o : classifier_multiplex_gmhmm.cpp
	$(MKDIR_P) ../../build/
	$(CC) $(CFLAGS) classifier_multiplex_gmhmm.cpp -o ../../build/classifier_multiplex_gmhmm.o

../../build/classifier_k_ary.o : classifier_k_ary.cpp
	$(MKDIR_P) ../../build/
	$(CC) $(CFLAGS) classifier_k_ary.cpp -o ../../build/classifier_k_ary.o

.PHONY : clean
clean :
	-rm -f ../../build/*.o *~ classifier &> /dev/null
