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

/*
 The Gaussian mixture hidden Markov model (GMHMM) is the essential building block of the full-scale model.
 
 It is an extension of the hidden Markov model (HMM) to continuous output spaces. It is primarily designed 
 for handling temporal data sets (i.e. observations of how some features change over time---a great usage 
 example is speech recognition), however it may also be used to represent any observations which can
 be sensibly ordered in some way.
 
 The implementation here has subsequently been extended to accept a Distribution pointer on construction,
 allowing for distributions other than the Gaussian to be specified and trained on the outputs.
*/

// A simple data structure storing the parameters used by the Baum-Welch algorithm
struct baumwelch_params
{
    int iterations; // maximal number of iterations
    double tolerance; // maximal allowed tolerance before convergence is assumed
};

std::istream& operator>>(std::istream &in, baumwelch_params &bw_p);
std::ostream& operator<<(std::ostream &out, const baumwelch_params bw_p);

class GMHMM
{
private:
    int n, obs; // number of nodes and sub-outputs
    double *pi; // start-state probability vector
    double **T; // transition probability matrix
    double **O; // sub-output emission matrix
    Distribution *d; // output distribution used for this layer (Gaussian by default)
    
public:
    GMHMM(int n, int obs); // initialise a random GMHMM
    GMHMM(int n, int obs, double *pi, double **T, double **O, double *mu, double *sigma); // load a known GMHMM from parameters
    GMHMM(int n, int obs, double *pi, double **T, double **O, Distribution *d); // load a known MHMM (with a custom output distribution) from parameters
    GMHMM(GMHMM *gmhmm); // copy an existing (G)MHMM
    ~GMHMM();

    /*
     Forward algorithm
     
     Input:      An observation sequence Y of length T.
     Output:     A triplet (alpha, c, L), where
                    - alpha(t, x) is the probability of producing the first
                      t elements of Y, and ending up in state x;
                    - c is a vector of scaling coefficients used at each step,
                      such that for any t', sum_x alpha(t', x) = 1 holds;
                    - L is the (log-)likelihood of producing sequence Y.
     Complexity: O(T * n^2) time, O(T * n) memory
    */
    std::tuple<double**, double*, double> forward(std::vector<std::pair<int, double> > &Y);
    
    /*
     Backward algorithm
     
     Input:      An observation sequence Y of length T, scaling coefficients c
     Output:     A matrix beta, where beta(t, x) is the likelihood of producing the
                 output elements Y[t+1], Y[t+2], ... Y[T], assuming we start from x.
                 The entries are scaled at each t using the given scaling coefficients.
     Complexity: O(T * n^2) time, O(T * n) memory
    */
    double** backward(std::vector<std::pair<int, double> > &Y, double *c);
    
    /*
     Baum-Welch algorithm
     
     Input:      Observation sequences Ys of combined length T, the maximal number of iterations to make,
                 and the tolerance to change (at smaller changes, convergence is assumed).
     Output:     A reevaluation of the model's parameters, such that it is more likely to produce Ys.
     Complexity: O(T * (n^2 + nm)) time (per iteration), O(T * n) memory
    */
    void baumwelch(std::vector<std::vector<std::pair<int, double> > > &Ys, int iterations, double tolerance);
    
    // Getters
    double get_pi(int x);
    double get_T(int i, int j);
    double get_O(int x, int y);
    Distribution* get_D();
    
    // Train the model parameters from a given training set
    void train(std::vector<std::vector<std::pair<int, double> > > &train_set, baumwelch_params &params);
    // Evaluate the log-likelihood of producing a given sequence (just runs the forward algorithm)
    double log_likelihood(std::vector<std::pair<int, double> > &test_data);

    // I/O operator overloads
    friend std::istream& operator>>(std::istream &in, GMHMM *&G);
    friend std::ostream& operator<<(std::ostream &out, const GMHMM *G);
};

#endif
