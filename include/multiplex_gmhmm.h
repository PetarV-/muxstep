/*
 Petar 'PetarV' Velickovic
 Data Structure: Multiplex GMHMM
*/

#ifndef MULTIPLEX_GMHMM
#define MULTIPLEX_GMHMM

#include <functional>
#include <vector>

#include <gmhmm.h>

class MultiplexGMHMM
{
private:
    int n;   // number of nodes in each layer 
    int obs; // number of sub-outputs
    int L;   // number of layers 
    std::vector<GMHMM*> layers; // the layers themselves
    double **omega; // the interlayer transition probability matrix
    
    std::vector<std::function<double(std::vector<double>)> > objectives;
public:
    MultiplexGMHMM(int n, int obs, int L); // initialise a random multiplex GMHMM
    MultiplexGMHMM(int n, int obs, int L, std::vector<GMHMM*> layers, double **omega); // initialise a multiplex GMHMM from parameters
    MultiplexGMHMM(int n, int obs, int L, FILE *f); // read the multiplex GMHMM from a file
    MultiplexGMHMM(MultiplexGMHMM *m_gmhmm); // Copy constructor
    ~MultiplexGMHMM();

    void dump(FILE *f);
    
    void set_omega(double **omega);
    void train(std::vector<std::vector<std::vector<double> > > &train_set);
    double log_likelihood(std::vector<std::vector<double> > &test_data);
    
    void dump_muxviz_data(char *nodes_filename, char *base_layers_filename);
};

#endif
