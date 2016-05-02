#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <algorithm>
#include <deque>
#include <queue>
#include <stack>
#include <set>
#include <map>
#include <complex>
#include <tuple>
#include <chrono>
#include <random>

#include <distribution.h>
#include <gaussian.h>
#include <gmhmm.h>

#define DPRINTC(C) printf(#C " = %c\n", (C))
#define DPRINTS(S) printf(#S " = %s\n", (S))
#define DPRINTD(D) printf(#D " = %d\n", (D))
#define DPRINTLLD(LLD) printf(#LLD " = %lld\n", (LLD))
#define DPRINTLF(LF) printf(#LF " = %.5lf\n", (LF))

using namespace std;
typedef unsigned int uint;
typedef long long lld;
typedef unsigned long long llu;

// define a RNG for U(0, 1)
default_random_engine gen;
uniform_real_distribution<double> rnd_real(0.0, 1.0);

istream& operator>>(istream &in, baumwelch_params &bw_p)
{
    in >> bw_p.iterations >> bw_p.tolerance;
    return in;
}

ostream& operator<<(ostream &out, const baumwelch_params bw_p)
{
    out << bw_p.iterations << " " << bw_p.tolerance << endl;
    return out;
}

// initialise a random GMHMM
GMHMM::GMHMM(int n, int obs) : n(n), obs(obs)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    gen = default_random_engine(seed);
    
    double total = 0.0;
    this -> pi = new double[n];
    for (int i=0;i<n;i++)
    {
        this -> pi[i] = rnd_real(gen);
        total += this -> pi[i];
    }
    for (int i=0;i<n;i++)
    {
        this -> pi[i] /= total;
    }
    
    this -> T = new double*[n];
    for (int i=0;i<n;i++)
    {
        this -> T[i] = new double[n];
        total = 0.0;
        for (int j=0;j<n;j++)
        {
            this -> T[i][j] = rnd_real(gen);
            total += this -> T[i][j];
        }
        for (int j=0;j<n;j++)
        {
            this -> T[i][j] /= total;
        }
    }
    
    this -> O = new double*[n];
    for (int i=0;i<n;i++)
    {
        this -> O[i] = new double[obs];
        total = 0.0;
        for (int j=0;j<obs;j++)
        {
            this -> O[i][j] = rnd_real(gen);
            total += this -> O[i][j];
        }
        for (int j=0;j<n;j++)
        {
            this -> O[i][j] /= total;
        }
    }
    
    this -> d = new Gaussian(obs);
}

// load a known GMHMM from parameters
GMHMM::GMHMM(int n, int obs, double *pi, double **T, double **O, double *mu, double *sigma) : n(n), obs(obs)
{
    this -> pi = new double[n];
    for (int i=0;i<n;i++) this -> pi[i] = pi[i];
    
    this -> T = new double*[n];
    for (int i=0;i<n;i++)
    {
        this -> T[i] = new double[n];
        for (int j=0;j<n;j++)
        {
            this -> T[i][j] = T[i][j];
        }
    }
    
    this -> O = new double*[n];
    for (int i=0;i<n;i++)
    {
        this -> O[i] = new double[obs];
        for (int j=0;j<obs;j++)
        {
            this -> O[i][j] = O[i][j];
        }
    }
    
    this -> d = new Gaussian(obs, mu, sigma);
}

// load a known MHMM (with a custom output distribution) from parameters
GMHMM::GMHMM(int n, int obs, double *pi, double **T, double **O, Distribution *d) : n(n), obs(obs)
{
    this -> pi = new double[n];
    for (int i=0;i<n;i++) this -> pi[i] = pi[i];
    
    this -> T = new double*[n];
    for (int i=0;i<n;i++)
    {
        this -> T[i] = new double[n];
        for (int j=0;j<n;j++)
        {
            this -> T[i][j] = T[i][j];
        }
    }
    
    this -> O = new double*[n];
    for (int i=0;i<n;i++)
    {
        this -> O[i] = new double[obs];
        for (int j=0;j<obs;j++)
        {
            this -> O[i][j] = O[i][j];
        }
    }
    
    this -> d = d -> clone();
}

// copy an existing (G)MHMM
GMHMM::GMHMM(GMHMM *gmhmm) : n(gmhmm -> n), obs(gmhmm -> obs)
{
    this -> pi = new double[gmhmm -> n];
    for (int i=0;i<gmhmm->n;i++) this -> pi[i] = gmhmm -> pi[i];
    
    this -> T = new double*[gmhmm -> n];
    for (int i=0;i<gmhmm->n;i++)
    {
        this -> T[i] = new double[gmhmm -> n];
        for (int j=0;j<gmhmm->n;j++)
        {
            this -> T[i][j] = gmhmm -> T[i][j];
        }
    }
    
    this -> O = new double*[gmhmm -> n];
    for (int i=0;i<gmhmm->n;i++)
    {
        this -> O[i] = new double[gmhmm -> obs];
        for (int j=0;j<gmhmm->obs;j++)
        {
            this -> O[i][j] = gmhmm -> O[i][j];
        }
    }
    
    this -> d = gmhmm -> d -> clone();
}

GMHMM::~GMHMM()
{
    delete[] pi;
    
    for (int i=0;i<n;i++)
    {
        delete[] T[i];
        delete[] O[i];
    }
    delete[] T;
    delete[] O;
    
    delete d;
}

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
tuple<double**, double*, double> GMHMM::forward(vector<pair<int, double> > &Y)
{
    int Ti = Y.size();
    
    // Initialise alpha
    double **alpha = new double*[Ti];
    for (int i=0;i<Ti;i++)
    {
        alpha[i] = new double[n];
    }
    double *c = new double[Ti];
    
    // Base case: alpha(0, i) = pi(i) * O(i, Y[0])
    double sum = 0.0;
    for (int i=0;i<n;i++)
    {
        alpha[0][i] = pi[i] * O[i][Y[0].first] * d -> get_probability(Y[0].first, Y[0].second);
        sum += alpha[0][i];
    }
    // Normalise to avoid underflow, keep track of scaling coefficient
    c[0] = 1.0 / sum;
    for (int i=0;i<n;i++)
    {
        alpha[0][i] /= sum;
    }
    
    for (int t=1;t<Ti;t++)
    {
        sum = 0.0;
        // Recurrence relation: alpha(t + 1, i) = O(i, Y[t + 1]) * sum_j {alpha(t, j) * T(j, i)}
        for (int i=0;i<n;i++)
        {
            alpha[t][i] = 0.0;
            for (int j=0;j<n;j++)
            {
                alpha[t][i] += alpha[t-1][j] * T[j][i];
            }
            alpha[t][i] *= O[i][Y[t].first] * d -> get_probability(Y[t].first, Y[t].second);
            sum += alpha[t][i];
        }
        
        // Normalise at each step to avoid underflow, keep track of scaling coefficients
        c[t] = 1.0 / sum;
        for (int i=0;i<n;i++)
        {
            alpha[t][i] /= sum;
        }
    }
    
    // Compute the log-likelihood of producing Y using the scaling coefficients
    double log_L = 0.0;
    for (int i=0;i<Ti;i++) log_L -= log(c[i]);
    
    return make_tuple(alpha, c, log_L);
}

/*
 Backward algorithm
 
 Input:      An observation sequence Y of length T, scaling coefficients c
 Output:     A matrix beta, where beta(t, x) is the likelihood of producing the
             output elements Y[t+1], Y[t+2], ... Y[T], assuming we start from x.
             The entries are scaled at each t using the given scaling coefficients.
 Complexity: O(T * n^2) time, O(T * n) memory
*/
double** GMHMM::backward(vector<pair<int, double> > &Y, double *c)
{
    int Ti = Y.size();
    
    // Initialise beta
    double **beta = new double*[Ti];
    for (int i=0;i<Ti;i++)
    {
        beta[i] = new double[n];
    }
    // Base case: beta(T - 1, i) = 1
    for (int i=0;i<n;i++) beta[Ti-1][i] = 1.0;
    
    for (int t=Ti-2;t>=0;t--)
    {
        for (int i=0;i<n;i++)
        {
            beta[t][i] = 0.0;
            // Recurrence relation: beta(t, i) = sum_j T(i, j) * O(j, Y[t + 1]) * beta(t + 1, j)
            for (int j=0;j<n;j++)
            {
                beta[t][i] += T[i][j] * O[j][Y[t+1].first] * d -> get_probability(Y[t+1].first, Y[t+1].second) * beta[t+1][j];
            }
            // Re-use scaling coefficients of alpha, to make the Baum-Welch iteration cleaner
            beta[t][i] *= c[t+1];
        }
    }
    
    return beta;
}

/*
 Baum-Welch algorithm
 
 Input:      Observation sequences Ys of combined length T, the maximal number of iterations to make,
             and the tolerance to change (at smaller changes, convergence is assumed).
 Output:     A reevaluation of the model's parameters, such that it is more likely to produce Ys.
 Complexity: O(T * (n^2 + nm)) time (per iteration), O(T * n) memory
*/
void GMHMM::baumwelch(vector<vector<pair<int, double> > > &Ys, int iterations, double tolerance)
{
    double ***alpha = new double**[Ys.size()];
    double ***beta = new double**[Ys.size()];
    double **c = new double*[Ys.size()];
    
    double PP, QQ;
    
    double lhood = 0.0;
    double oldlhood = 0.0;
    
    for (int iter=0;iter<iterations;iter++)
    {
        lhood = 0.0;
        
        // Run the forward-backward algorithm for each individual sequence
        for (uint l=0;l<Ys.size();l++)
        {
            tuple<double**, double*, double> x = forward(Ys[l]);
            alpha[l] = get<0>(x);
            c[l] = get<1>(x);
            lhood += get<2>(x);
            beta[l] = backward(Ys[l], c[l]);
        }
        
        double **nextO = new double*[n];
        for (int i=0;i<n;i++) nextO[i] = new double[obs];
        
        // Reestimate the model parameters (using the EM iteration)
        for (int i=0;i<n;i++)
        {
            // pi(i) = alpha(0, i) * beta(0, i)
            // (summed over all sequences, and normalised over all i)
            pi[i] = 0.0;
            for (uint l=0;l<Ys.size();l++)
            {
                pi[i] += alpha[l][0][i] * beta[l][0][i];
            }
            pi[i] /= Ys.size();
            
            QQ = 0.0;
            
            /*
             The further steps will compute the (normalised) parameters:
             
              gamma(t, i) = alpha(t, i) * beta(t, i)
              xi(t, i, j) = c(t + 1) * alpha(t, i) * T(i, j) * O(j, Y[t + 1]) * beta(t + 1, j)
             
             and use them to reestimate the model parameters T and O, as follows:
                T(i, j) = sum_t xi(t, i, j) / sum_t gamma(t, i)
                O(i, y) = sum_t I(Y[t] = y) * gamma(t, i) / sum_t gamma(t, i)
             
             N.B. xi is not directly computed, in order to save memory.
            */
            
            for (int k=0;k<obs;k++)
            {
                nextO[i][k] = 0.0;
            }
            
            for (uint l=0;l<Ys.size();l++)
            {
                for (uint t=0;t<Ys[l].size()-1;t++)
                {
                    double curr = alpha[l][t][i] * beta[l][t][i];
                    QQ += curr;
                    nextO[i][Ys[l][t].first] += curr;
                }
            }
            
            for (int j=0;j<n;j++)
            {
                PP = 0.0;
                for (uint l=0;l<Ys.size();l++)
                {
                    for (uint t=0;t<Ys[l].size()-1;t++)
                    {
                        PP += alpha[l][t][i] * O[j][Ys[l][t+1].first] * d -> get_probability(Ys[l][t+1].first, Ys[l][t+1].second) * beta[l][t+1][j] * c[l][t+1];
                    }
                }
                T[i][j] *= PP / QQ;
            }
            
            for (uint l=0;l<Ys.size();l++)
            {
                int lim = Ys[l].size() - 1;
                double curr = alpha[l][lim][i] * beta[l][lim][i];
                QQ += curr;
                nextO[i][Ys[l][lim].first] += curr;
            }
            
            for (int k=0;k<obs;k++)
            {
                nextO[i][k] /= QQ;
            }
        }
        
        // Remove all temporaries
        for (uint l=0;l<Ys.size();l++)
        {
            for (uint t=0;t<Ys[l].size();t++)
            {
                delete[] alpha[l][t];
                delete[] beta[l][t];
            }
            delete[] alpha[l];
            delete[] beta[l];
            delete[] c[l];
        }
        
        for (int i=0;i<n;i++)
        {
            delete[] O[i];
        }
        delete[] O;
        
        O = nextO;
        
        // Check whether the algorithm has converged in likelihood wrt the tolerance
        if (fabs(lhood - oldlhood) < tolerance) break;
        
        oldlhood = lhood;
    }
}

double GMHMM::get_pi(int x)
{
    return this -> pi[x];
}

double GMHMM::get_T(int i, int j)
{
    return this -> T[i][j];
}

double GMHMM::get_O(int x, int y)
{
    return this -> O[x][y];
}

Distribution* GMHMM::get_D()
{
    return this -> d;
}

// Re(randomise) the model parameters
void GMHMM::reset()
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    gen = default_random_engine(seed);
    
    double total = 0.0;
    for (int i=0;i<n;i++)
    {
        this -> pi[i] = rnd_real(gen);
        total += this -> pi[i];
    }
    for (int i=0;i<n;i++)
    {
        this -> pi[i] /= total;
    }
    
    for (int i=0;i<n;i++)
    {
        total = 0.0;
        for (int j=0;j<n;j++)
        {
            this -> T[i][j] = rnd_real(gen);
            total += this -> T[i][j];
        }
        for (int j=0;j<n;j++)
        {
            this -> T[i][j] /= total;
        }
    }
    
    for (int i=0;i<n;i++)
    {
        total = 0.0;
        for (int j=0;j<obs;j++)
        {
            this -> O[i][j] = rnd_real(gen);
            total += this -> O[i][j];
        }
        for (int j=0;j<n;j++)
        {
            this -> O[i][j] /= total;
        }
    }
}

// Train the model parameters from a given training set
void GMHMM::train(vector<vector<pair<int, double> > > &train_set, baumwelch_params &params)
{
    // train the distribution parameters
    d -> train(train_set);
    
    // now run the Baum-Welch algorithm
    baumwelch(train_set, params.iterations, params.tolerance);
}

// Evaluate the log-likelihood of producing a given sequence (just runs the forward algorithm)
double GMHMM::log_likelihood(vector<pair<int, double> > &test_data)
{
    tuple<double**, double*, double> x = forward(test_data);
    double **alpha = get<0>(x);
    double *c = get<1>(x);
    
    // Delete the temporaries to avoid memory leaks
    for (uint i=0;i<test_data.size();i++)
    {
        delete[] alpha[i];
    }
    delete[] alpha;
    delete[] c;
    
    // Report only the (log-)likelihood
    return get<2>(x);
}

// Read a GMHMM from a given input stream
istream& operator>>(istream &in, GMHMM *&G)
{
    if (G != NULL) delete G;
    
    int n, obs;
    in >> n >> obs;

    double *pi = new double[n];
    for (int i=0;i<n;i++)
    {
        in >> pi[i];
    }

    double **T = new double*[n];
    for (int i=0;i<n;i++)
    {
        T[i] = new double[n];
        for (int j=0;j<n;j++)
        {
            in >> T[i][j];
        }
    }

    double **O = new double*[n];
    for (int i=0;i<n;i++)
    {
        O[i] = new double[obs];
        for (int j=0;j<obs;j++)
        {
            in >> O[i][j];
        }
    }
    
    Distribution *d;
    if (G == NULL) d = new Gaussian(obs);
    else
    {
        d = G -> d;
        delete G;
    }
    
    d -> read(in);

    G = new GMHMM(n, obs, pi, T, O, d);

    return in;
}

// Write a GMHMM to a given output stream
ostream& operator<<(ostream &out, const GMHMM *G)
{
    out << G -> n << " " << G -> obs << endl;

    for (int i=0;i<G->n;i++)
    {
        out << G -> pi[i] << " ";
    }
    out << endl;

    for (int i=0;i<G->n;i++)
    {
        for (int j=0;j<G->n;j++)
        {
            out << G -> T[i][j] << " ";
        }
        out << endl;
    }

    for (int i=0;i<G->n;i++)
    {
        for (int j=0;j<G->obs;j++)
        {
            out << G -> O[i][j] << " ";
        }
        out << endl;
    }

    G -> d -> write(out);

    return out;
}

