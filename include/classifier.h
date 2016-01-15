#ifndef CLASSIFIER
#define CLASSIFIER

#include <iostream>
#include <vector>

#include <multiplex_gmhmm.h>

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

class MultiplexGMHMMClassifier : public Classifier<std::vector<std::pair<int, std::vector<double> > >, bool>
{
private:
    int node_count; // the number of nodes in each layer
    int sub_count;  // the number of sub-outputs
    int type_count; // the number of data types
    MultiplexGMHMM *positive_model;
    MultiplexGMHMM *negative_model;
    
    std::vector<double> thresholds;
    
public:
    MultiplexGMHMMClassifier(int node_count, int sub_count, int type_count); // initialise a random multiplex GMHMM
    MultiplexGMHMMClassifier(int node_count, int sub_count, int type_count, MultiplexGMHMM *positive, MultiplexGMHMM *negative); // initialise a multiplex GMHMM from parameters 
    ~MultiplexGMHMMClassifier();

    Classifier<std::vector<std::pair<int, std::vector<double> > >, bool>* clone();
    
    void dump_muxviz(char *positive_nodes_filename, char *positive_base_layers_filename, char *negative_nodes_filename, char *negative_base_layers_filename); // dump the positive and negative models into a format readable by muxViz

    void train(std::vector<std::pair<std::vector<std::pair<int, std::vector<double> > >, bool> > &training_set);
    bool classify(std::vector<std::pair<int, std::vector<double> > > &test_data);
    
    std::vector<double> get_thresholds();

    friend std::istream& operator>>(std::istream &in, MultiplexGMHMMClassifier *&C);
    friend std::ostream& operator<<(std::ostream &out, const MultiplexGMHMMClassifier *C);
};

#endif
