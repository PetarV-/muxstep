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
#include <chrono>
#include <random>

#include <nsga2.h>

#define EPS 1e-9
#define INF 987654321

#define DPRINTC(C) printf(#C " = %c\n", (C))
#define DPRINTS(S) printf(#S " = %s\n", (S))
#define DPRINTD(D) printf(#D " = %d\n", (D))
#define DPRINTLLD(LLD) printf(#LLD " = %lld\n", (LLD))
#define DPRINTLF(LF) printf(#LF " = %.5lf\n", (LF))

using namespace std;
typedef unsigned int uint;
typedef long long lld;
typedef unsigned long long llu;

bool NSGAII::cmp_by_key(const chromosome &X, const chromosome &Y)
{
    return (X.values[X.sort_key] < Y.values[Y.sort_key]);
}

bool NSGAII::is_equal(chromosome &X, chromosome &Y)
{
    for (int i=0;i<ft_size;i++)
    {
        if (fabs(X.features[i] - Y.features[i]) > EPS) return false;
    }
    return true;
}

bool NSGAII::dominated_by(chromosome &X, chromosome &Y)
{
    for (int i=0;i<obj_size;i++)
    {
        if (X.values[i] > Y.values[i]) return false;
    }
    return true;
}

int NSGAII::get_ft_size()
{
    return ft_size;
}

int NSGAII::get_obj_size()
{
    return obj_size;
}

list<chromosome> NSGAII::find_nondominated_front(vector<chromosome> &P)
{
    list<chromosome> ret;
    ret.push_back(P[0]);
    for (uint i=1;i<P.size();i++)
    {
        chromosome p = P[i];
        bool add_p = true;
        list<chromosome>::iterator it = ret.begin();
        while (it != ret.end())
        {
            chromosome q = (*it);
            if (dominated_by(p, q)) ret.erase(it++);
            else if (dominated_by(q, p)) { add_p = false; break; }
            else it++;
        }
        if (add_p) ret.push_back(p);
    }
    return ret;
}

vector<vector<chromosome> > NSGAII::fast_nondominated_sort(vector<chromosome> &P)
{
    vector<vector<chromosome> > F;
    
    vector<vector<int> > Sp;
    vector<int> np;
    vector<int> Q;
    Sp.resize(P.size());
    np.resize(P.size());
    for (uint i=0;i<P.size();i++)
    {
        chromosome p = P[i];
        Sp[i].clear();
        np[i] = 0;
        for (uint j=0;j<P.size();j++)
        {
            chromosome q = P[j];
            if (dominated_by(p, q)) Sp[i].push_back(j);
            else if (dominated_by(q, p)) np[i]++;
        }
        if (np[i] == 0) Q.push_back(i);
    }
    
    vector<chromosome> Fi;
    Fi.resize(Q.size());
    for (uint i=0;i<Q.size();i++)
    {
        Fi[i] = P[Q[i]];
        Fi[i].rank = 1;
    }
    F.push_back(Fi);
    
    int ii = 1;
    while (!Q.empty())
    {
        vector<int> R;
        for (uint i=0;i<Q.size();i++)
        {
            for (uint j=0;j<Sp[Q[i]].size();j++)
            {
                int q = Sp[Q[i]][j];
                if (--np[q] == 0) R.push_back(q);
            }
        }
        ii++;
        if (R.empty()) break;
        
        Fi.resize(R.size());
        for (uint i=0;i<R.size();i++)
        {
            Fi[i] = P[R[i]];
            Fi[i].rank = ii;
        }
        F.push_back(Fi);
        Q = R;
    }
    
    P.clear();
    return F;
}

void NSGAII::crowding_distance_assignment(vector<chromosome> &I)
{
    int l = I.size();
    for (int i=0;i<l;i++) I[i].distance = 0;
    for (int obj=0;obj<obj_size;obj++)
    {
        for (int i=0;i<l;i++) I[i].sort_key = obj;
        sort(I.begin(), I.end(), cmp_by_key);
        I[0].distance = I[l-1].distance = INF;
        for (int i=1;i<l-1;i++)
        {
            I[i].distance += I[i+1].values[obj] - I[i-1].values[obj];
        }
    }
}

void NSGAII::initialise()
{
    main_population.clear();
    for (int i=0;i<pop_size;i++)
    {
        chromosome curr;
        curr.features.resize(ft_size);
        curr.values.resize(obj_size);
        
        for (int j=0;j<ft_size;j++)
        {
            curr.features[j] = var_lims[j].first + rand_real(generator) * (var_lims[j].second - var_lims[j].first);
        }
        
        for (int obj=0;obj<obj_size;obj++)
        {
            curr.values[obj] = objectives[obj](curr.features);
        }
        
        main_population.push_back(curr);
    }
}

int NSGAII::select(vector<chromosome> &P)
{
    int i1 = rand_index(generator);
    int i2 = rand_index(generator);
    
    chromosome cand_1 = P[i1];
    chromosome cand_2 = P[i2];
    
    if (cand_1 < cand_2) return i1;
    else return i2;
}

pair<chromosome, chromosome> NSGAII::crossover(chromosome &P1, chromosome &P2)
{
    if (rand_real(generator) <= p_crossover)
    {
        chromosome C1, C2;
        C1.features.resize(ft_size);
        C2.features.resize(ft_size);
        C1.values.resize(obj_size);
        C2.values.resize(obj_size);
        
        for (int i=0;i<ft_size;i++)
        {
            double par1 = P1.features[i];
            double par2 = P2.features[i];
            
            double lo = var_lims[i].first;
            double hi = var_lims[i].second;
            
            if (rand_real(generator) <= 0.5)
            {
                if (fabs(par1 - par2) > 1e-6)
                {
                    double v1, v2, alpha, beta, betaq;
                    
                    if (par2 > par1) v1 = par1, v2 = par2;
                    else v1 = par2, v2 = par1;
                    
                    if ((v1 - lo) > (hi - v2)) beta = 1 + (2*(hi - v2)/(v2 - v1));
                    else beta = 1 + (2*(v1 - lo)/(v2 - v1));
                    
                    double step = di_crossover + 1.0;
                    beta = 1.0 / beta;
                    alpha = 2.0 - pow(beta, step);
                    
                    assert(alpha >= 0.0);
                    
                    double rnd = rand_real(generator);
                    
                    if (rnd <= 1.0 / alpha)
                    {
                        alpha *= rnd;
                        step = 1.0 / (di_crossover + 1.0);
                        betaq = pow(alpha, step);
                    }
                    else
                    {
                        alpha = 1.0 / (2.0 - (alpha * rnd));
                        step = 1.0 / (di_crossover + 1.0);
                        assert(alpha >= 0.0);
                        betaq = pow(alpha, step);
                    }
                    
                    C1.features[i] = 0.5 * ((v1 + v2) - betaq*(v2 - v1));
                    C2.features[i] = 0.5 * ((v1 + v2) + betaq*(v2 - v1));
                }
                else
                {
                    double betaq = 1.0;
                    double v1 = par1, v2 = par2;
                    C1.features[i] = 0.5 * ((v1 + v2) - betaq*(v2 - v1));
                    C2.features[i] = 0.5 * ((v1 + v2) + betaq*(v2 - v1));
                }
                
                if (C1.features[i] < lo) C1.features[i] = lo;
                if (C1.features[i] > hi) C1.features[i] = hi;
                if (C2.features[i] < lo) C2.features[i] = lo;
                if (C2.features[i] > hi) C2.features[i] = hi;
            }
            else
            {
                C1.features[i] = par1;
                C2.features[i] = par2;
            }
        }
            
        return make_pair(C1, C2);
    }
    else
    {
        chromosome C1 = P1;
        chromosome C2 = P2;
        return make_pair(C1, C2);
    }
}

void NSGAII::mutate(vector<chromosome> &P)
{
    for (int i=0;i<pop_size;i++)
    {
        for (int j=0;j<ft_size;j++)
        {
            if (rand_real(generator) <= p_mutation)
            {
                double val = P[pop_size + i].features[j];
                double lo = var_lims[j].first;
                double hi = var_lims[j].second;
                
                double delta, deltaq;
                
                if (val > lo)
                {
                    if ((val - lo) < (hi - val)) delta = (val - lo)/(hi - lo);
                    else delta = (hi - val)/(hi - lo);
                    
                    double rnd = rand_real(generator);
                    
                    double indi = 1.0 / (di_mutation + 1.0);
                    
                    if (rnd <= 0.5)
                    {
                        double xy = 1.0 - delta;
                        double v = 2*rnd + (1 - 2 * rnd) * (pow(xy, (di_mutation + 1)));
                        deltaq = pow(v, indi) - 1.0;
                    }
                    else
                    {
                        double xy = 1.0 - delta;
                        double v = 2 * (1 - rnd) + 2 * (rnd - 0.5) * (pow(xy, (di_mutation + 1)));
                        deltaq = 1.0 - pow(v, indi);
                    }
                    
                    val += deltaq * (hi - lo);
                    if (val < lo) val = lo;
                    if (val > hi) val = hi;
                    P[pop_size + i].features[j] = val;
                }
                else
                {
                    P[pop_size + i].features[j] = lo + rand_real(generator) * (hi - lo);
                }
            }
        }
    }
}

void NSGAII::make_new_pop(vector<chromosome> &P)
{
    for (int i=0;i<pop_size >> 1;i++)
    {
        chromosome p1, p2;
        p1 = P[select(P)];
        p2 = P[select(P)];
        pair<chromosome, chromosome> Cret = crossover(p1, p2);
        P.push_back(Cret.first);
        P.push_back(Cret.second);
    }
    mutate(P); // mutates only new generation, leaves first N alone
    for (int i=0;i<pop_size;i++)
    {
        for (int obj=0;obj<obj_size;obj++)
        {
            P[pop_size + i].values[obj] = objectives[obj](P[pop_size + i].features);
        }
    }
}

void NSGAII::iterate()
{
    make_new_pop(main_population);
    vector<vector<chromosome> > fronts = fast_nondominated_sort(main_population);
    int ii = 0;
    while (main_population.size() + fronts[ii].size() <= uint(pop_size))
    {
        crowding_distance_assignment(fronts[ii]);
        main_population.insert(main_population.end(), fronts[ii].begin(), fronts[ii].end());
        ii++;
    }
    int elements_needed = pop_size - main_population.size();
    if (elements_needed > 0)
    {
        crowding_distance_assignment(fronts[ii]);
        sort(fronts[ii].begin(), fronts[ii].end());
        main_population.insert(main_population.end(), fronts[ii].begin(), fronts[ii].begin() + elements_needed);
    }
}

vector<chromosome> NSGAII::optimise(nsga2_params &params, vector<function<double(vector<double>)> > &objs)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator = default_random_engine(seed);
    objectives = objs;
    
    pop_size = params.pop_size;
    ft_size = params.ft_size;
    obj_size = params.obj_size;
    generations = params.generations;
    p_crossover = params.p_crossover;
    p_mutation = params.p_mutation;
    di_crossover = params.di_crossover;
    di_mutation = params.di_mutation;
    var_lims = params.var_lims;
    
    rand_index = uniform_int_distribution<int>(0, pop_size - 1);
    
    printf("Input successfully read. Initialising...\n");
    
    initialise();
    
    printf("Initialised zeroth generation.\n");
    
    for (int i=0;i<generations;i++)
    {
        iterate();
        if ((i+1)%20 == 0) printf("Generation %d complete.\n", i+1);
    }
    
    printf("Done.\n");
    
    return main_population;
}
