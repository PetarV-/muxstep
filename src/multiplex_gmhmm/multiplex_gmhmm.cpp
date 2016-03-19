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
#include <functional>

#include <multiplex_gmhmm.h>
#include <nsga2.h>

#define DPRINTC(C) printf(#C " = %c\n", (C))
#define DPRINTS(S) printf(#S " = %s\n", (S))
#define DPRINTD(D) printf(#D " = %d\n", (D))
#define DPRINTLLD(LLD) printf(#LLD " = %lld\n", (LLD))
#define DPRINTLF(LF) printf(#LF " = %.5lf\n", (LF))

using namespace std;
typedef unsigned int uint;
typedef long long lld;
typedef unsigned long long llu;

// initialise a random multiplex GMHMM
MultiplexGMHMM::MultiplexGMHMM(int n, int obs, int L) : n(n), obs(obs), L(L)
{
    this -> layers.resize(L);
    for (int i=0;i<L;i++)
    {
        this -> layers[i] = new GMHMM(n, obs);
    }
    
    this -> omega = new double*[L];
    for (int i=0;i<L;i++)
    {
        this -> omega[i] = new double[L];
        for (int j=0;j<L;j++)
        {
            this -> omega[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

// initialise a multiplex GMHMM from parameters
MultiplexGMHMM::MultiplexGMHMM(int n, int obs, int L, vector<GMHMM*> layers, double **omega) : n(n), obs(obs), L(L)
{
    this -> layers.resize(L);
    for (int i=0;i<L;i++)
    {
        this -> layers[i] = new GMHMM(layers[i]);
    }

    this -> omega = new double*[L];
    for (int i=0;i<L;i++)
    {
        this -> omega[i] = new double[L];
        for (int j=0;j<L;j++)
        {
            this -> omega[i][j] = omega[i][j];
        }
    }
}

// Copy constructor
MultiplexGMHMM::MultiplexGMHMM(MultiplexGMHMM *m_gmhmm) : n(m_gmhmm -> n), obs(m_gmhmm -> obs), L(m_gmhmm -> L)
{
    layers.resize(m_gmhmm -> L);
    for (int i=0;i<m_gmhmm->L;i++)
    {
        layers[i] = new GMHMM(m_gmhmm -> layers[i]);
    }

    omega = new double*[m_gmhmm -> L];
    for (int i=0;i<m_gmhmm->L;i++)
    {
        omega[i] = new double[m_gmhmm -> L];
        for (int j=0;j<m_gmhmm->L;j++)
        {
            omega[i][j] = m_gmhmm -> omega[i][j];
        }
    }
}

MultiplexGMHMM::~MultiplexGMHMM()
{
    for (int i=0;i<L;i++) delete layers[i];
    for (int i=0;i<L;i++) delete[] omega[i];
    delete[] omega;
}

// Setter for the interlayer transition matrix; useful while training
void MultiplexGMHMM::set_omega(double **omega)
{
    for (int i=0;i<L;i++)
    {
        double sum = 0.0;
        for (int j=0;j<L;j++)
        {
            this -> omega[i][j] = omega[i][j];
            sum += omega[i][j];
        }
        // Perform normalisation here, just in case
        for (int j=0;j<L;j++)
        {
            this -> omega[i][j] /= sum;
        }
    }
}

// Train the model parameters from a given training set
void MultiplexGMHMM::train(vector<vector<pair<int, vector<double> > > > &train_set, nsga2_params &nsga_p, baumwelch_params &bw_p)
{
    // Train all the layers individually (as before)
    for (int l=0;l<L;l++)
    {
        vector<vector<pair<int, double> > > curr_set(train_set.size());
        for (uint i=0;i<train_set.size();i++)
        {
            curr_set[i].resize(train_set[i].size());
            for (uint j=0;j<train_set[i].size();j++)
            {
                curr_set[i][j].first = train_set[i][j].first;
                curr_set[i][j].second = train_set[i][j].second[l];
            }
        }
        layers[l] -> train(curr_set, bw_p);
    }
    
    // Define the lambdas that calculate likelihoods for a given omega
    objectives.resize(train_set.size());
    for (uint t=0;t<train_set.size();t++)
    {
        objectives[t] = [this, t, &train_set] (vector<double> X) -> double
        {
            // Convert the parameters in X into an omega-matrix
            double **temp_omega = new double*[L];
            for (int i=0;i<L;i++)
            {
                temp_omega[i] = new double[L];
                for (int j=0;j<L;j++)
                {
                    temp_omega[i][j] = X[i*L + j];
                }
            }
            
            // Set it as the current omega for the model
            set_omega(temp_omega);
            
            for (int i=0;i<L;i++) delete[] temp_omega[i];
            delete[] temp_omega;
            
            // Evaluate the log-likelihood on this training sequence
            return -log_likelihood(train_set[t]);
        };
    }
    
    // These NSGA-II parameters are model/training set dependent and have to be fixed here
    nsga_p.ft_size = L * L;
    nsga_p.obj_size = train_set.size();
    
    // Run the NSGA-II algorithm
    NSGAII nsga2;
    vector<chromosome> candidates = nsga2.optimise(nsga_p, objectives);
    
    // Evaluate the best choice of omega out of the ones returned
    // Here we discriminate by favouring more sequences with good likelihoods
    // rather than fewer sequences with superb likelihoods.
    int best = -1;
    double min_sum = -1.0;
    for (uint i=0;i<candidates.size();i++)
    {
        sort(candidates[i].values.begin(), candidates[i].values.end());
        double curr_sum = 0.0;
        for (uint j=0;j<train_set.size();j++)
        {
            curr_sum += (j + 1) * candidates[i].values[j];
        }
        if (best == -1 || curr_sum < min_sum)
        {
            best = i;
            min_sum = curr_sum;
        }
    }
    
    // Adjust the parameters accordingly
    double **fin_omega = new double*[L];
    for (int i=0;i<L;i++)
    {
        fin_omega[i] = new double[L];
        for (int j=0;j<L;j++)
        {
            fin_omega[i][j] = candidates[best].features[i*L + j];
        }
    }
    
    set_omega(fin_omega);
    
    for (int i=0;i<L;i++) delete[] fin_omega[i];
    delete[] fin_omega;
}

// Evaluate the log-likelihood of producing a given sequence (just runs the forward algorithm)
double MultiplexGMHMM::log_likelihood(vector<pair<int, vector<double> > > &test_data)
{
    double ret = 0.0;
    
    double ***A = new double**[test_data.size()];
    for (uint i=0;i<test_data.size();i++)
    {
        A[i] = new double*[L];
        for (int j=0;j<L;j++)
        {
            A[i][j] = new double[n];
        }
    }
    
    // Detailed comments about the forward algorithm iteration may be already found
    // in src/gmhmm/gmhmm.cpp, so they are omitted here.
    
    double *pi = new double[L];
    double pi_sum = 0.0;
    for (int i=0;i<L;i++) pi_sum += omega[i][i];
    for (int i=0;i<L;i++) pi[i] = omega[i][i] / pi_sum;
    
    int first_sub = test_data[0].first;
    double init_sum = 0.0;
    for (int i=0;i<L;i++)
    {
        double curr_type_val = test_data[0].second[i];
        double curr_prob = layers[i] -> get_D() -> get_probability(first_sub, curr_type_val);
        for (int j=0;j<n;j++)
        {
            double curr_pi = layers[i] -> get_pi(j);
            double curr_o = layers[i] -> get_O(j, first_sub);
            A[0][i][j] = pi[i] * curr_pi * curr_o * curr_prob;
            init_sum += A[0][i][j];
        }
    }
    
    for (int i=0;i<L;i++)
    {
        for (int j=0;j<n;j++)
        {
            A[0][i][j] /= init_sum;
        }
    }
    ret += log(init_sum);
    
    for (uint t=1;t<test_data.size();t++)
    {
        double fullsum = 0.0;
        int curr_sub = test_data[t].first;
        
        for (int i=0;i<L;i++)
        {
            double curr_type_val = test_data[t].second[i];
            double curr_prob = layers[i] -> get_D() -> get_probability(curr_sub, curr_type_val);
            
            for (int j=0;j<n;j++)
            {
                double sum = 0.0;
                double curr_g = layers[i] -> get_O(j, curr_sub);
                
                for (int ii=0;ii<L;ii++)
                {
                    if (ii == i)
                    {
                        for (int jj=0;jj<n;jj++)
                        {
                            double curr_t = layers[i] -> get_T(jj, j);
                            sum += A[t-1][ii][jj] * omega[ii][i] * curr_t * curr_g * curr_prob;
                        }
                    }
                    else sum += A[t-1][ii][j] * omega[ii][i] * curr_g * curr_prob;
                }
                
                A[t][i][j] = sum;
                fullsum += sum;
            }
        }
        
        for (int i=0;i<L;i++)
        {
            for (int j=0;j<n;j++)
            {
                A[t][i][j] /= fullsum;
            }
        }
        
        ret += log(fullsum);
    }
    
    for (uint i=0;i<test_data.size();i++)
    {
        for (int j=0;j<L;j++)
        {
            delete[] A[i][j];
        }
        delete[] A[i];
    }
    delete[] A;
    delete[] pi;
    
    return ret;
}

// Read a Multiplex GMHMM from a given input stream
istream& operator>>(istream &in, MultiplexGMHMM *&M)
{
    if (M == NULL) delete M;

    int n, obs, L;
    in >> n >> obs >> L;

    vector<GMHMM*> layers;
    layers.resize(L);
    for (int i=0;i<L;i++)
    {
        in >> layers[i];
    }

    double **omega = new double*[L];
    for (int i=0;i<L;i++)
    {
        omega[i] = new double[L];
        for (int j=0;j<L;j++)
        {
            in >> omega[i][j];
        }
    }

    M = new MultiplexGMHMM(n, obs, L, layers, omega);

    return in;
}

// Write a multiplex GMHMM to a given output stream
ostream& operator<<(ostream &out, const MultiplexGMHMM *M)
{
    out << M -> n << " " << M -> obs << " " << M -> L << endl;

    for (int i=0;i<M->L;i++)
    {
        out << M -> layers[i];
    }

    for (int i=0;i<M->L;i++)
    {
        for (int j=0;j<M->L;j++)
        {
            out << M -> omega[i][j] << " ";
        }
        out << endl;
    }

    return out;
}

