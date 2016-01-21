OBJS = ../build/classifier_evaluator.o ../build/classifier_multiplex_gmhmm.o ../build/gmhmm.o ../build/multiplex_gmhmm.o ../build/nsga2.o ../build/phi.o ../build/basic_usage.o
CC = clang++ -I../include
DEBUG = -g
CFLAGS = -std=c++11 -O3 -fPIC -Wall -Wextra -Werror -Weffc++ -Wstrict-aliasing --pedantic -c $(DEBUG)
LFLAGS = -lpthread -Wall $(DEBUG)
MKDIR_P = mkdir -p

../bin/basic_usage : $(OBJS)
	$(MKDIR_P) ../../bin/
	$(CC) $(LFLAGS) $(OBJS) -o ../bin/basic_usage

../build/classifier_evaluator.o :
	$(MAKE) -C ../test/classifier ../../build/classifier_evaluator.o

../build/classifier_multiplex_gmhmm.o :
	$(MAKE) -C ../src/classifier/ ../../build/classifier_multiplex_gmhmm.o

../build/gmhmm.o :
	$(MAKE) -C ../src/gmhmm/ ../../build/gmhmm.o

../build/multiplex_gmhmm.o :
	$(MAKE) -C ../src/multiplex_gmhmm/ ../../build/multiplex_gmhmm.o

../build/nsga2.o :
	$(MAKE) -C ../src/nsga2/ ../../build/nsga2.o

../build/phi.o : 
	$(MAKE) -C ../src/gmhmm/ ../../build/phi.o

../build/basic_usage.o :
	$(MKDIR_P) ../build/
	$(CC) $(CFLAGS) basic_usage.cpp -o ../build/basic_usage.o

.PHONY : clean
clean :
	-rm -f ../bin/basic_usage *~ basic_usage &> /dev/null
