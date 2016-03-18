/*
 Petar 'PetarV' Velickovic
 Data Structure: Multiplex GMHMM
*/

#ifndef MULTIPLEX_GMHMM
#define MULTIPLEX_GMHMM

#include <functional>
#include <iostream>
#include <vector>

#include <gmhmm.h>
#include <nsga2.h>

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
    MultiplexGMHMM(MultiplexGMHMM *m_gmhmm); // Copy constructor
    ~MultiplexGMHMM();

    void set_omega(double **omega);
    void train(std::vector<std::vector<std::pair<int, std::vector<double> > > > &train_set, nsga2_params &nsga_p, baumwelch_params &bw_p);
    double log_likelihood(std::vector<std::pair<int, std::vector<double> > > &test_data);
    
    void dump_muxviz_data(char *nodes_filename, char *base_layers_filename);
    
    friend std::istream& operator>>(std::istream& in, MultiplexGMHMM *&M);
    friend std::ostream& operator<<(std::ostream& out, const MultiplexGMHMM *M);
};

#endif
