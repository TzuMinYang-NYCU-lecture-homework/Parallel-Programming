#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  for (int i = 0; i < numNodes; ++i)
  {
    solution[i] = equal_prob;
  }

  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
  double* temp_sol = (double*)malloc(sizeof(double) * g->num_nodes);
  int* out_size = (int*)malloc(sizeof(int) * g->num_nodes);
  //const Vertex* in_beg = (const Vertex*)malloc(sizeof(const Vertex) * g->num_nodes), * in_end = (const Vertex*)malloc(sizeof(const Vertex) * g->num_nodes);
  int num_no_outgoing = 0, * no_out_vertex = (int*)malloc(sizeof(int) * g->num_nodes);

  // compute all info avoid call function multiple times
  for (int i = 0; i < numNodes; ++i)
  {
    /*
    in_beg[i] = incoming_begin(g, i);
    in_end[i] = incoming_end(g, i);
    */
    out_size[i] = outgoing_size(g, i);
    if (out_size[i] == 0)
    {
      no_out_vertex[num_no_outgoing++] = i;
    }
  }

  double global_diff;
  bool converged = false;
  while (!converged)
  {
    // compute score_new[vi] for all nodes vi:

    // compute sum over all nodes v in graph with no outgoing edges { damping * score_old[v] / numNodes }
    double sum_no_outgoing = 0.0;
    #pragma omp parallel for reduction(+:sum_no_outgoing)
    for (int i = 0; i < num_no_outgoing; ++i)
    {
      sum_no_outgoing += damping * solution[no_out_vertex[i]] / numNodes;
    }

    #pragma omp parallel for
    for (int i = 0; i < numNodes; ++i)
    {
      // initialize score_new[vi] = 0
      temp_sol[i] = 0;

      // score_new[vi] = sum over all nodes vj reachable from incoming edges { score_old[vj] / number of edges leaving vj  }
      for (const Vertex* j = incoming_begin(g, i); j != incoming_end(g, i); j++)
      {
        temp_sol[i] += solution[*j] / out_size[*j];
      }

      // score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;
      temp_sol[i] = (damping * temp_sol[i]) + (1.0 - damping) / numNodes;

      // score_new[vi] += sum over all nodes v in graph with no outgoing edges { damping * score_old[v] / numNodes }
      temp_sol[i] += sum_no_outgoing;
    }

    // compute how much per-node scores have changed
    // quit once algorithm has converged

    //global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
    global_diff = 0;
    #pragma omp parallel for reduction (+:global_diff)
    for (int i = 0; i < numNodes; ++i)
    {
      global_diff += abs(temp_sol[i] - solution[i]);
    }

    // update solution
    for (int i = 0; i < numNodes; ++i)
    {
      solution[i] = temp_sol[i];
    }

    // converged = (global_diff < convergence)
    //printf("%f\n", global_diff);
    converged = (global_diff < convergence);
  }

}
