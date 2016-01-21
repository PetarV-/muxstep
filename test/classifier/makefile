OBJS = ../../build/classifier_evaluator.o ../../build/classifier_multiplex_gmhmm.o ../../build/gmhmm.o ../../build/multiplex_gmhmm.o ../../build/nsga2.o ../../build/phi.o ../../build/simple_tester.o
CC = clang++ -I../../include
DEBUG = -g
CFLAGS = -std=c++11 -O3 -fPIC -Wall -Wextra -Werror -Weffc++ -Wstrict-aliasing --pedantic -c $(DEBUG)
LFLAGS = -lpthread -Wall $(DEBUG)
MKDIR_P = mkdir -p

../../bin/simple_tester : $(OBJS)
	$(MKDIR_P) ../../bin/
	$(CC) $(LFLAGS) $(OBJS) -o ../../bin/simple_tester

../../build/classifier_multiplex_gmhmm.o :
	$(MAKE) -C ../../src/classifier/ ../../build/classifier_multiplex_gmhmm.o

../../build/gmhmm.o :
	$(MAKE) -C ../../src/gmhmm/ ../../build/gmhmm.o

../../build/multiplex_gmhmm.o :
	$(MAKE) -C ../../src/multiplex_gmhmm/ ../../build/multiplex_gmhmm.o

../../build/nsga2.o :
	$(MAKE) -C ../../src/nsga2/ ../../build/nsga2.o

../../build/phi.o : 
	$(MAKE) -C ../../src/gmhmm/ ../../build/phi.o

../../build/simple_tester.o :
	$(MKDIR_P) ../../build/
	$(CC) $(CFLAGS) simple_tester.cpp -o ../../build/simple_tester.o

../../build/classifier_evaluator.o : classifier_evaluator.cpp
	$(MKDIR_P) ../../build/
	$(CC) $(CFLAGS) classifier_evaluator.cpp -o ../../build/classifier_evaluator.o

.PHONY : clean
clean :
	-rm -f ../../build/*.o *~ classifier_evaluator &> /dev/null
	-rm -f ../../bin/simple_tester *~ simple_tester &> /dev/null
