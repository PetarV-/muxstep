OBJS = build/classifier_multiplex_gmhmm.o build/classifier_k_ary.o build/gmhmm.o build/multiplex_gmhmm.o build/nsga2.o build/phi.o
CC = clang++ -Iinclude
LFLAGS = -lpthread -Wall
SFLAGS = -shared -fPIC
MKDIR_P = mkdir -p

.PHONY : all
all : static shared

.PHONY : static
static : $(OBJS)
	$(MKDIR_P) lib/
	ar rcs lib/libmuxstep.a $(OBJS)

.PHONY : shared
shared : $(OBJS)
	$(MKDIR_P) lib/
	$(CC) $(SFLAGS) $(LFLAGS) -o lib/libmuxstep.dyn.so $(OBJS)

build/classifier_multiplex_gmhmm.o :
	$(MAKE) -C src/classifier/ ../../build/classifier_multiplex_gmhmm.o

build/classifier_k_ary.o :
	$(MAKE) -C src/classifier/ ../../build/classifier_k_ary.o

build/gmhmm.o :
	$(MAKE) -C src/gmhmm/ ../../build/gmhmm.o

build/multiplex_gmhmm.o :
	$(MAKE) -C src/multiplex_gmhmm ../../build/multiplex_gmhmm.o

build/nsga2.o :
	$(MAKE) -C src/nsga2 ../../build/nsga2.o

build/phi.o :
	$(MAKE) -C src/gmhmm ../../build/phi.o

.PHONY : clean
clean :
	rm -f build/*.o &> /dev/null
	rm -f lib/*.a &> /dev/null
	rm -f lib/*.so &> /dev/null

