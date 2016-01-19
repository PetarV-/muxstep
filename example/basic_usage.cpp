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
#include <tuple>
#include <fstream>

#include <classifier.h>
#include <classifier_evaluator.h>

#define DPRINTC(C) printf(#C " = %c\n", (C))
#define DPRINTS(S) printf(#S " = %s\n", (S))
#define DPRINTD(D) printf(#D " = %d\n", (D))
#define DPRINTLLD(LLD) printf(#LLD " = %lld\n", (LLD))
#define DPRINTLF(LF) printf(#LF " = %.5lf\n", (LF))

using namespace std;
typedef unsigned int uint;
typedef long long lld;
typedef unsigned long long llu;

int main()
{
     MultiplexGMHMMClassifier *C = new MultiplexGMHMMClassifier(4, 4, 2);
     ifstream in("model.txt");
     in >> C;
     in.close();

    // Use the trained classifier to make predictions
    // (Here using data produced by syn_gen with the same parameters)
    auto data_tst = get<2>(extract_data("syn_test.out"));
    int total_correct = 0, total = 0;
    for (auto it = data_tst.begin(); it != data_tst.end(); it++)
    {
        bool prediction = C -> classify(it -> first);
        bool expected = it -> second;
        cout << "Predicted class is: " << (prediction ? "positive" : "negative") << ". ";
        cout << ((prediction == expected) ? "Correct!" : "Incorrect!") << endl;
        total++; 
        if (prediction == expected) total_correct++;
    }

    double accuracy = (total_correct * 1.0) / (total * 1.0);
    cout << "Accuracy: " << accuracy << endl;

    // The model could be stored for later use (perhaps into a file)
    ofstream out("model.txt");
    out << C;
    out.close();

    return 0;
}
