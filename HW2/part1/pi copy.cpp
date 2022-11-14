#include <cstdio>
#include <pthread.h>
#include <cstdlib>
#include <ctime>

using namespace std;

long long number_in_circle  = 0;

pthread_mutex_t mutex;

static unsigned long a=123456789, b=362436069, c=521288629;

unsigned long xorshf96(void) {          //period 2^96-1
unsigned long t;
    a ^= a << 16;
    a ^= a >> 5;
    a ^= a << 1;

   t = a;
   a = b;
   b = c;
   c = t ^ a ^ b;

  return c;
}

void *estimate_pi(void *input_info)
{
    long long toss_num = *((int *)input_info);

    double x = 0, y = 0;
    long long local_count = 0;

    for(long long i = 0; i < toss_num; i++)
    {
        // (double)rand() / RAND_MAX可產生[0, 1]的小數
        // rand內部實作有lock
        x = (double)xorshf96() / 123456789.0 * 2 + (-1);
        y = (double)xorshf96() / 123456789.0 * 2 + (-1);

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
    long long input_to_func[thread_num];

    for(int i = 0; i < thread_num ; i++)
    {
        if(i == thread_num - 1) input_to_func[i] = point_num / thread_num + point_num % thread_num; //最後一個要多負責處理那些除不盡的point數
        else input_to_func[i] = point_num / thread_num;

        pthread_create(&thread[i % thread_num], NULL, estimate_pi, (void *)&input_to_func[i]);
    }

    for(int i = 0; i < thread_num ; i++) pthread_join(thread[i], NULL);

	printf("%lf\n", 4.0 * (double)number_in_circle  / (double)point_num);

    return 0;
}