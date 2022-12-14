#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
// add by myself
#include <set>
//

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
//#define VERBOSE
#define THRESHOLD 10000000

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void hybrid_top_down_step(
    Graph g,
    int cur_dis,
    int *count,
    int *distances)
{
    int local_count = 0, new_dis = cur_dis + 1;
    #pragma omp parallel for schedule(static, 1) reduction(+: local_count) //!!! reduction can't parse frontier->count
    for (int i = 0; i < g->num_nodes; ++i)
    {   
        if (distances[i] == cur_dis)
        {
            int start_edge = g->outgoing_starts[i];
            int end_edge = (i == g->num_nodes - 1)
                            ? g->num_edges
                            : g->outgoing_starts[i + 1];
            
            for (int neighbor = start_edge; neighbor < end_edge; ++neighbor)
            {
                int outgoing = g->outgoing_edges[neighbor];
                if (distances[outgoing] == NOT_VISITED_MARKER)
                {
                    distances[outgoing] = new_dis;
                    ++local_count;
                }
            }
        }
    }
    *count = local_count;           
}


// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
    // add by myself
    #pragma omp parallel for
    //
    for (int i = 0; i < frontier->count; i++)
    {
        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // add by myself
        //int local_count = 0;
        //int local_frontier[g->num_nodes];
        //std::vector<int> local_frontier;
        std::set<int> local_frontier;
        //

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];

            /*if (distances[outgoing] == NOT_VISITED_MARKER)
            {
                distances[outgoing] = distances[node] + 1;
                int index = new_frontier->count++;
                new_frontier->vertices[index] = outgoing;
            }*/

            // add by myself
            if (distances[outgoing] == NOT_VISITED_MARKER)
            {
                distances[outgoing] = distances[node] + 1;
                //local_frontier[local_count++] = outgoing;
                //local_frontier.push_back(outgoing);
                local_frontier.insert(outgoing);
            }                                                                                                                                                                                        
            //                                                                                                                                                                                                                  
        }     
        
        // add by myself
        // type __sync_fetch_and_add (type *ptr, type value, ...)
        //int cur_count = __sync_fetch_and_add(&(new_frontier->count), local_count);
        int cur_count = __sync_fetch_and_add(&(new_frontier->count), local_frontier.size());
        //for (int j = 0; j < local_frontier.size(); ++j)
        //    new_frontier->vertices[cur_count + j] = local_frontier[j];
        for (const auto &s : local_frontier)
            new_frontier->vertices[cur_count++] = s;
        //
    }                                     
}


// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    // add by myself
    #pragma omp parallel for
    //
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

// add by myself
void bottom_up_step(
    Graph g, 
    int cur_dis,
    int *count,
    int *distances)
{
    int local_count = 0, new_dis = cur_dis + 1;
    #pragma omp parallel for schedule(static, 1) reduction(+: local_count) //!!! reduction can't parse frontier->count
    for (int i = 0; i < g->num_nodes; ++i)
    {   
        if (distances[i] == NOT_VISITED_MARKER)
        {
            int start_edge = g->incoming_starts[i];
            int end_edge = (i == g->num_nodes - 1)
                            ? g->num_edges
                            : g->incoming_starts[i + 1];
            
            for (int neighbor = start_edge; neighbor < end_edge; ++neighbor)
            {
                int incoming = g->incoming_edges[neighbor];
                if (distances[incoming] == cur_dis)
                {
                    distances[i] = new_dis;
                    ++local_count;
                    break;
                }
            }
        }
    }
    *count = local_count;
}
//

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    // add by myself
    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    sol->distances[ROOT_NODE_ID] = 0;

    int count = 1, cur_dis = 0;
    while (count != 0)
    {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        bottom_up_step(graph, cur_dis++, &count, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", count, end_time - start_time);
#endif
    }
    //
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    
    // add by myself

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    sol->distances[ROOT_NODE_ID] = 0;

    int count = 1, cur_dis = 0;
    while (count != 0)
    {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        if(count >= THRESHOLD)
            bottom_up_step(graph, cur_dis++, &count, sol->distances);
        else
            hybrid_top_down_step(graph, cur_dis++, &count, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", count, end_time - start_time);
#endif
    }
    //
}
