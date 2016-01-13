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
    ~GMHMM();
    
    std::tuple<double**, double*, double> forward(std::vector<std::pair<double, int> > &Y);
    double** backward(std::vector<std::pair<double, int> > &Y, double *c);
    void baumwelch(std::vector<std::vector<double> > &Ys, int iterations, double tolerance);
    void baumwelch(std::vector<std::vector<std::pair<double, int> > > &sorted_Ys, int iterations, double tolerance); // sorted
    
    double get_pi(int x);
    double get_T(int i, int j);
    double get_O(int x, int y);
    double get_probability(int obs_id, double x);
    void train(std::vector<std::vector<double> > &train_set);
    void train(std::vector<std::vector<std::pair<double, int> > > &train_set); // sorted
    double log_likelihood(std::vector<double> &test_data);
    double log_likelihood(std::vector<std::pair<double, int> > &sorted_data); // sorted
};

#endif
