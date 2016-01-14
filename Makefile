CXX=g++
CXXFLAGS=-Wall -O3 -std=c++0x
LDFLAGS=-lm -lcityhash
BIN=tdbtg_preorderer_train tdbtg_preorderer_parse

all : $(BIN)

clean : 
	rm -f *.o $(BIN)

tdbtg_preorderer.o : tdbtg_preorderer.h mini_hash_map.h

tdbtg_preorderer_train : tdbtg_preorderer_train.o tdbtg_preorderer.o
	$(CXX) $(CXXFLAGS) -o $@ $+ $(LDFLAGS)

tdbtg_preorderer_parse : tdbtg_preorderer_parse.o tdbtg_preorderer.o
	$(CXX) $(CXXFLAGS) -o $@ $+ $(LDFLAGS)
