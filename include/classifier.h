/*
 Petar 'PetarV' Velickovic
 Data Structure: Multiplex GMHMM Classifier
*/

#ifndef CLASSIFIER
#define CLASSIFIER

#include <iostream>
#include <vector>

#include <gmhmm.h>
#include <multiplex_gmhmm.h>
#include <nsga2.h>

template<typename Data, typename Label>
class Classifier
{
public:
    virtual ~Classifier() { }
    virtual Classifier* clone() = 0; // Needed by the evaluator (for parallel crossvalidation)
    virtual void train(std::vector<std::pair<Data, Label> > &training_set) = 0;
    virtual Label classify(Data &test_data) = 0;
    virtual std::vector<double> get_thresholds() = 0; // Needed by the evaluator (for extracting ROC data)
};

// the standard binary classifier
class MultiplexGMHMMClassifier : public Classifier<std::vector<std::pair<int, std::vector<double> > >, bool>
{
private:
    int node_count; // the number of nodes in each layer
    int sub_count;  // the number of sub-outputs
    int type_count; // the number of data types
    MultiplexGMHMM *positive_model;
    MultiplexGMHMM *negative_model;
    
    nsga2_params nsga_p;
    baumwelch_params bw_p;
    
    std::vector<double> thresholds;
    
public:
    MultiplexGMHMMClassifier(int node_count, int sub_count, int type_count, nsga2_params nsga_p, baumwelch_params bw_p); // initialise a random multiplex GMHMM classifier
    MultiplexGMHMMClassifier(int node_count, int sub_count, int type_count, nsga2_params nsga_p, baumwelch_params bw_p, MultiplexGMHMM *positive, MultiplexGMHMM *negative); // initialise a multiplex GMHMM classifier from parameters
    ~MultiplexGMHMMClassifier();

    Classifier<std::vector<std::pair<int, std::vector<double> > >, bool>* clone();
    
    void dump_muxviz(char *positive_nodes_filename, char *positive_base_layers_filename, char *negative_nodes_filename, char *negative_base_layers_filename); // dump the positive and negative models into a format readable by muxViz

    void train(std::vector<std::pair<std::vector<std::pair<int, std::vector<double> > >, bool> > &training_set);
    bool classify(std::vector<std::pair<int, std::vector<double> > > &test_data);
    std::pair<bool, bool> classify_reliable(std::vector<std::pair<int, std::vector<double> > > &test_data, double min_margin);
    
    std::vector<double> get_thresholds();

    friend std::istream& operator>>(std::istream &in, MultiplexGMHMMClassifier *&C);
    friend std::ostream& operator<<(std::ostream &out, const MultiplexGMHMMClassifier *C);
};

// Experimental (untested) feature: k-ary classifier
class MultiplexKClassifier : public Classifier<std::vector<std::pair<int, std::vector<double> > >, int>
{
private:
    int node_count;
    int sub_count;
    int type_count;
    int label_count; // how many labels do we have?
    std::vector<MultiplexGMHMM*> models;
    
    nsga2_params nsga_p;
    baumwelch_params bw_p;

public:
    MultiplexKClassifier(int node_count, int sub_count, int type_count, int label_count, nsga2_params nsga_p, baumwelch_params bw_p); // initialise a random k-ary classifier
    MultiplexKClassifier(int node_count, int sub_count, int type_count, int label_count, nsga2_params nsga_p, baumwelch_params bw_p, std::vector<MultiplexGMHMM*> models); // initialise a k-ary classifier from parameters
    ~MultiplexKClassifier();

    Classifier<std::vector<std::pair<int, std::vector<double> > >, int>* clone();

    void train(std::vector<std::pair<std::vector<std::pair<int, std::vector<double> > >, int> > &training_set);
    int classify(std::vector<std::pair<int, std::vector<double> > > &test_data);
    std::pair<int, bool> classify_reliable(std::vector<std::pair<int, std::vector<double> > > &test_data, double min_margin);

    std::vector<double> get_thresholds(); // unnecessary here, returns empty vector

    friend std::istream& operator>>(std::istream &in, MultiplexKClassifier *&C);
    friend std::ostream& operator<<(std::ostream &out, const MultiplexKClassifier *C);
};

#endif
