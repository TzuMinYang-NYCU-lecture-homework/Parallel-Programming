#include <cstdio>
#include <pthread.h>
#include <cstdlib>
#include <ctime>

using namespace std;

struct arg
{
    long long count;
    int id;
};

long long number_in_circle  = 0;

pthread_mutex_t mutex;

void *estimate_pi(void *input_info)
{
    struct arg temp = *((struct arg *)input_info);
    long long toss_num = temp.count;
    unsigned int seed = (temp.id + 5) * 3;

    double x = 0, y = 0;
    long long local_count = 0;

    for(long long i = 0; i < toss_num; i++)
    {
        // (double)rand_r(&seed) / RAND_MAX可產生[0, 1]的小數
        // rand內部實作有lock
        x = (double)rand_r(&seed) / (double)RAND_MAX * 2.0 + (-1.0);
        y = (double)rand_r(&seed) / (double)RAND_MAX * 2.0 + (-1.0);

        if(x * x + y * y <= 1) local_count++;
    }

    pthread_mutex_lock(&mutex);
    number_in_circle  += local_count;
    pthread_mutex_unlock(&mutex);
    
    pthread_exit(NULL);
} 

int main (int argc, const char * argv[]) 
{
    int thread_num = strtoll(argv[1], NULL, 10);
    long long point_num = strtoll(argv[2], NULL, 10);

    //每個thread分配一些point去算
    pthread_t thread[thread_num];
    struct arg input_to_func[thread_num];

    for(int i = 0; i < thread_num ; i++)
    {
        if(i == thread_num - 1) input_to_func[i].count = point_num / thread_num + point_num % thread_num; //最後一個要多負責處理那些除不盡的point數
        else input_to_func[i].count = point_num / thread_num;

        input_to_func[i].id = i;
        pthread_create(&thread[i % thread_num], NULL, estimate_pi, (void *)&input_to_func[i]);
    }

    for(int i = 0; i < thread_num ; i++) pthread_join(thread[i], NULL);

	printf("%lf\n", 4.0 * (double)number_in_circle  / (double)point_num);

    return 0;
}