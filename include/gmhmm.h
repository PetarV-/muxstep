/*
 Petar 'PetarV' Velickovic
 Data Structure: (Gaussian) Mixture Hidden Markov Model
*/

#ifndef GM_HIDDEN_MARKOV_MODEL
#define GM_HIDDEN_MARKOV_MODEL

#include <iostream>
#include <tuple>
#include <vector>

#include <distribution.h>

struct baumwelch_params
{
    int iterations;
    double tolerance;
};

std::istream& operator>>(std::istream &in, baumwelch_params &bw_p);
std::ostream& operator<<(std::ostream &out, const baumwelch_params bw_p);

class GMHMM
{
private:
    int n, obs; // number of nodes and "sub-observations"
    double *pi; // start-state probability vector
    double **T; // transition probability matrix
    double **O; // sub-output emission matrix
    Distribution *d; // output distribution used for this layer (Gaussian by default)
    
public:
    GMHMM(int n, int obs); // initialise a random GMHMM
    GMHMM(int n, int obs, double *pi, double **T, double **O, double *mu, double *sigma); // load a known GMHMM
    GMHMM(int n, int obs, double *pi, double **T, double **O, Distribution *d); // load a known MHMM (with a custom output distribution)
    GMHMM(GMHMM *gmhmm); // copy an existing GMHMM
    ~GMHMM();

    std::tuple<double**, double*, double> forward(std::vector<std::pair<int, double> > &Y);
    double** backward(std::vector<std::pair<int, double> > &Y, double *c);
    void baumwelch(std::vector<std::vector<std::pair<int, double> > > &Ys, int iterations, double tolerance);
    
    double get_pi(int x);
    double get_T(int i, int j);
    double get_O(int x, int y);
    Distribution* get_D();

    void train(std::vector<std::vector<std::pair<int, double> > > &train_set, baumwelch_params &params);
    double log_likelihood(std::vector<std::pair<int, double> > &test_data);

    friend std::istream& operator>>(std::istream &in, GMHMM *&G);
    friend std::ostream& operator<<(std::ostream &out, const GMHMM *G);
};

#endif
