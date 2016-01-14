/*
 Petar 'PetarV' Velickovic
 Data Structure: Gaussian Mixture Hidden Markov Model
*/

#ifndef GM_HIDDEN_MARKOV_MODEL
#define GM_HIDDEN_MARKOV_MODEL

#include <tuple>
#include <vector>

// helper function; Phi(x; mean, stddev)
double gaussian_pdf(double x, double mean, double stdev);

class GMHMM
{
private:
    int n, obs; // number of nodes and "sub-observations"
    double *pi; // start-state probability vector
    double **T; // transition probability matrix
    double **O; // sub-output emission matrix
    double *mu, *sigma; // means and variances for each sub-output
    
public:
    GMHMM(int n, int obs); // initialise a random GMHMM
    GMHMM(int n, int obs, double *pi, double **T, double **O, double *mu, double *sigma); // load a known GMHMM
    GMHMM(int n, int obs, FILE *f); // load a GMHMM from a file
    GMHMM(GMHMM *gmhmm); // copy an existing GMHMM
    ~GMHMM();

    void dump(FILE *f);
    
    std::tuple<double**, double*, double> forward(std::vector<std::pair<int, double> > &Y);
    double** backward(std::vector<std::pair<int, double> > &Y, double *c);
    void baumwelch(std::vector<std::vector<std::pair<int, double> > > &Ys, int iterations, double tolerance);
    
    double get_pi(int x);
    double get_T(int i, int j);
    double get_O(int x, int y);
    double get_probability(int obs_id, double x);
    void train(std::vector<std::vector<std::pair<int, double> > > &train_set);
    double log_likelihood(std::vector<std::pair<int, double> > &test_data);
};

#endif
