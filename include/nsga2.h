/*
 Petar 'PetarV' Velickovic
 Algorithm: NSGA-II
*/

#ifndef NSGA_II
#define NSGA_II

#include <functional>
#include <random>
#include <vector>

/*
 The NSGA-II algorithm, as described by Deb et al. (2002), is a multiobjective
 genetic algorithm that attempts to find an a set of Pareto-optimal solutions
 to a problem of optimising several functions simultaneously.
*/

// A single candidate solution (commonly known as a chromosome)
struct chromosome
{
    int rank; // ID of nondominated front
    double distance; // crowding distance
    int sort_key; // for sorting purposes
    
    std::vector<double> features; // actual components of the solution
    std::vector<double> values; // values of the objective functions
    
    // Operator for ordering chromosomes by desirability
    bool operator <(const chromosome &c) const
    {
        if (rank != c.rank) return (rank < c.rank);
        else return (distance > c.distance);
    }
};

// A structure for storing the parameters of the algorithm
struct nsga2_params
{
    int pop_size; // Population size (#chromosomes)
    int ft_size; // Features size (#variables)
    int obj_size; // Objectives size (#functions to optimise)
    int generations; // Number of generations to create (#iterations)
    
    double p_crossover; // Crossover probability
    double p_mutation; // Mutation probability
    double di_crossover; // Crossover distribution parameter
    double di_mutation; // Mutation distribution parameter
    
    std::vector<std::pair<double, double> > var_lims; // upper and lower bounds of variables
};


std::istream& operator>>(std::istream &in, nsga2_params &nsga_p);
std::ostream& operator<<(std::ostream &out, const nsga2_params nsga_p);

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
    
    // RNGs used by the algorithm
    std::default_random_engine generator;
    std::uniform_int_distribution<int> rand_index;
    std::uniform_real_distribution<double> rand_real;
    
    std::vector<std::pair<double, double> > var_lims;
    std::vector<std::function<double(std::vector<double>)> > objectives;
    
    // The current population vector of candidate chromosomes
    std::vector<chromosome> main_population;
    
public:
    // Helper boolean comparators of two chromosomes
    static bool cmp_by_key(const chromosome &X, const chromosome &Y);
    bool is_equal(chromosome &X, chromosome &Y);
    bool dominated_by(chromosome &X, chromosome &Y);
    
    // Getters
    int get_ft_size();
    int get_obj_size();

    // Partition the chromosomes into nondominated fronts
    std::vector<std::vector<chromosome> > fast_nondominated_sort(std::vector<chromosome> &P);

    // Assign crowding distances to chromosomes within a single nondominated front
    void crowding_distance_assignment(std::vector<chromosome> &I);
    
    // Initialise the population
    void initialise();
    
    // Select a chromosome for crossover
    int select(std::vector<chromosome> &P);
    // Perform crossover between two chromosomes, producing two children chromosomes
    std::pair<chromosome, chromosome> crossover(chromosome &P1, chromosome &P2);
    // Perfrom a mutation on a chromosome
    void mutate(std::vector<chromosome> &P);
    // Generate the next generation of chromosomes
    void make_new_pop(std::vector<chromosome> &P);
    
    // Perform a single full iteration of NSGA-II
    void iterate();

    // Run the NSGA-II algorithm for given parameters and objective functions
    std::vector<chromosome> optimise(nsga2_params &params, std::vector<std::function<double(std::vector<double>)> > &objs);
};

#endif
