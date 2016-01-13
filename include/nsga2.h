/*
 Petar 'PetarV' Velickovic
 Algorithm: NSGA-II
*/

#ifndef NSGA_II
#define NSGA_II

#include <functional>
#include <random>
#include <vector>

struct chromosome
{
    int rank;
    double distance;
    int sort_key; // for sorting purposes
    std::vector<double> features;
    std::vector<double> values;
    
    bool operator <(const chromosome &c) const
    {
        if (rank != c.rank) return (rank < c.rank);
        else return (distance > c.distance);
    }
};

struct nsga2_params
{
    int pop_size;
    int ft_size;
    int obj_size;
    int generations;
    
    double p_crossover;
    double p_mutation;
    double di_crossover;
    double di_mutation;
    
    std::vector<std::pair<double, double> > var_lims;
};

class NSGAII
{
private:
    int pop_size;
    int ft_size;
    int obj_size;
    int generations;
    
    double p_crossover;
    double p_mutation;
    double di_crossover;
    double di_mutation;
    
    std::default_random_engine generator;
    std::uniform_int_distribution<int> rand_index;
    std::uniform_real_distribution<double> rand_real;
    
    std::vector<std::pair<double, double> > var_lims;
    std::vector<std::function<double(std::vector<double>)> > objectives;
    
    std::vector<chromosome> main_population;
    
public:
    static bool cmp_by_key(const chromosome &X, const chromosome &Y);
    bool is_equal(chromosome &X, chromosome &Y);
    bool dominated_by(chromosome &X, chromosome &Y);
    
    int get_ft_size();
    int get_obj_size();

    std::list<chromosome> find_nondominated_front(std::vector<chromosome> &P);

    std::vector<std::vector<chromosome> > fast_nondominated_sort(std::vector<chromosome> &P);

    void crowding_distance_assignment(std::vector<chromosome> &I);
    
    void initialise();
    
    int select(std::vector<chromosome> &P);
    std::pair<chromosome, chromosome> crossover(chromosome &P1, chromosome &P2);
    void mutate(std::vector<chromosome> &P);
    void make_new_pop(std::vector<chromosome> &P);
    
    void iterate();

    std::vector<chromosome> optimise(nsga2_params &params, std::vector<std::function<double(std::vector<double>)> > &objs);
};

#endif
