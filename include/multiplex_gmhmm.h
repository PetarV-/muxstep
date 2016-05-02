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

/*
 A Multiplex GMHMM consists of a series of GMHMM layers over the same set of nodes
 (where each layer can be specialised for producing its own data type), 
 interconnected with further interlayer transitions between images of a node in each layer.
 
 The full structure itself behaves like a larger-scale GMHMM, allowing us to reuse standard
 inference algorithms (e.g. the forward algorithm) on it.
*/

class MultiplexGMHMM
{
private:
    int n;   // number of nodes in each layer 
    int obs; // number of sub-outputs
    int L;   // number of layers 
    std::vector<GMHMM*> layers; // the layers themselves
    double **omega; // the interlayer transition probability matrix
    
    // the objective functions for NSGA-II
    std::vector<std::function<double(std::vector<double>)> > objectives;
public:
    MultiplexGMHMM(int n, int obs, int L); // initialise a random multiplex GMHMM
    MultiplexGMHMM(int n, int obs, int L, std::vector<GMHMM*> layers, double **omega); // initialise a multiplex GMHMM from parameters
    MultiplexGMHMM(MultiplexGMHMM *m_gmhmm); // Copy constructor
    ~MultiplexGMHMM();

    // Setter for the interlayer transition matrix; useful while training
    void set_omega(double **omega);

    // (Re)randomises the model parameters
    void reset();
    // Train the model parameters from a given training set
    void train(std::vector<std::vector<std::pair<int, std::vector<double> > > > &train_set, nsga2_params &nsga_p, baumwelch_params &bw_p);
    // Evaluate the log-likelihood of producing a given sequence (just runs the forward algorithm)
    double log_likelihood(std::vector<std::pair<int, std::vector<double> > > &test_data);
    
    // I/O operator overloads
    friend std::istream& operator>>(std::istream& in, MultiplexGMHMM *&M);
    friend std::ostream& operator<<(std::ostream& out, const MultiplexGMHMM *M);
};

#endif
