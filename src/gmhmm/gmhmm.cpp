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

tuple<double**, double*, double> GMHMM::forward(vector<pair<int, double> > &Y)
{
    int Ti = Y.size();
    
    double **alpha = new double*[Ti];
    for (int i=0;i<Ti;i++)
    {
        alpha[i] = new double[n];
    }
    double *c = new double[Ti];
    
    double sum = 0.0;
    for (int i=0;i<n;i++)
    {
        alpha[0][i] = pi[i] * O[i][Y[0].first] * d -> get_probability(Y[0].first, Y[0].second);
        sum += alpha[0][i];
    }
    c[0] = 1.0 / sum;
    for (int i=0;i<n;i++)
    {
        alpha[0][i] /= sum;
    }
    
    for (int t=1;t<Ti;t++)
    {
        sum = 0.0;
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
        
        c[t] = 1.0 / sum;
        for (int i=0;i<n;i++)
        {
            alpha[t][i] /= sum;
        }
    }
    
    double log_L = 0.0;
    for (int i=0;i<Ti;i++) log_L -= log(c[i]);
    
    return make_tuple(alpha, c, log_L);
}

double** GMHMM::backward(vector<pair<int, double> > &Y, double *c)
{
    int Ti = Y.size();
    
    double **beta = new double*[Ti];
    for (int i=0;i<Ti;i++)
    {
        beta[i] = new double[n];
    }
    for (int i=0;i<n;i++) beta[Ti-1][i] = 1.0;
    
    for (int t=Ti-2;t>=0;t--)
    {
        for (int i=0;i<n;i++)
        {
            beta[t][i] = 0.0;
            for (int j=0;j<n;j++)
            {
                beta[t][i] += T[i][j] * O[j][Y[t+1].first] * d -> get_probability(Y[t+1].first, Y[t+1].second) * beta[t+1][j];
            }
            beta[t][i] *= c[t+1];
        }
    }
    
    return beta;
}

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
        
        for (int i=0;i<n;i++)
        {
            pi[i] = 0.0;
            for (uint l=0;l<Ys.size();l++)
            {
                pi[i] += alpha[l][0][i] * beta[l][0][i];
            }
            pi[i] /= Ys.size();
            
            QQ = 0.0;
            
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

void GMHMM::train(vector<vector<pair<int, double> > > &train_set, baumwelch_params &params)
{
    // train the distribution parameters
    d -> train(train_set);

    // reset the initial probabilities
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
    
    // now run the Baum-Welch algorithm
    baumwelch(train_set, params.iterations, params.tolerance);
}

double GMHMM::log_likelihood(vector<pair<int, double> > &test_data)
{
    tuple<double**, double*, double> x = forward(test_data);
    double **alpha = get<0>(x);
    double *c = get<1>(x);
    
    for (uint i=0;i<test_data.size();i++)
    {
        delete[] alpha[i];
    }
    delete[] alpha;
    delete[] c;
    
    return get<2>(x);
}

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

