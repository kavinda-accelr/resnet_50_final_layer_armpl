#include "Thread_Pool.h"

#if __linux__ == 1
static int s_core_number = 0;

void Reset_Core_Counter()
{
    s_core_number = 0;
}

void Check_Multi_Thread_Compatibility()
{
    std::cout << "Hardware concurrency : " << std::thread::hardware_concurrency() << std::endl;
}

const int Schedule_Thread(std::thread& thread, const int core_number)
{
    int max_core_number = std::thread::hardware_concurrency();

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    const int core_no = (core_number > -1 && core_number < max_core_number) ? core_number
        : (s_core_number < max_core_number) ? s_core_number : 0;

    s_core_number = (core_number > -1 && core_number < max_core_number) ? s_core_number
        : (s_core_number < max_core_number) ? s_core_number + 1 : 0;

    CPU_SET(core_no, &cpuset);
    int rc = pthread_setaffinity_np(thread.native_handle(),
        sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }

    return core_no;
}

void Print_Thread_Info()
{
    std::cout << "Process thread id: " << std::this_thread::get_id() << std::endl;
    std::cout << "Processor Info core number : " << sched_getcpu() << std::endl;
}

#else
static int s_core_number = 0;

void Reset_Core_Counter()
{
    s_core_number = 0;
}

void Check_Multi_Thread_Compatibility()
{
    std::cout << "Hardware concurrency : " << std::thread::hardware_concurrency() << std::endl;
}

const int Schedule_Thread(std::thread& thread, const int core_number)
{
    int max_core_number = std::thread::hardware_concurrency();

    const int core_no = (core_number > -1 && core_number < max_core_number) ? core_number
        : (s_core_number < max_core_number) ? s_core_number : 0;

    DWORD_PTR mask = (static_cast<DWORD_PTR>(1) << core_no);

    s_core_number = (core_number > -1 && core_number < max_core_number) ? s_core_number
        : (s_core_number < max_core_number) ? s_core_number + 1 : 0;

    if (!SetThreadAffinityMask(thread.native_handle(), mask))
    {
        std::cerr << "Error calling SetThreadAffinityMask: " << GetLastError() << std::endl;
    }

    return core_no;
}
void Print_Thread_Info()
{
    std::cout << "Process thread id: " << std::this_thread::get_id() << std::endl;
    std::cout << "Processor Info core number : " << GetCurrentProcessorNumber() << std::endl;
}
#endif

#if THREAD_POOL_CV == 0
Thread_Pool::Thread_Pool(const unsigned int num_threads)
{
    _num_threads = (num_threads > 0) ? num_threads : std::thread::hardware_concurrency();
    for (unsigned int i = 0; i < _num_threads; i++)
    {
        _threads.emplace_back(std::thread(Thread_Pool::thread_work, this));
        const int core_no = Schedule_Thread(_threads[i]);
        _core_numbers.push_back(core_no);
    }
}

Thread_Pool::Thread_Pool(const Thread_Pool& thread_pool) :
    _num_threads(thread_pool.get_num_threads()),
    _core_numbers(thread_pool.get_core_numbers())
{
    for (unsigned int i = 0; i < _num_threads; i++)
    {
        _threads.emplace_back(std::thread(Thread_Pool::thread_work, this));
        const int core_no = Schedule_Thread(_threads[i], _core_numbers[i]);
        _core_numbers.push_back(core_no);
    }
}

void Thread_Pool::assign(std::function<void()> work)
{
    std::unique_lock<std::mutex> queue_lck(_queue_mutex);
    _work_queue.push(work);
    _queue_size++;
    queue_lck.unlock();
}

void Thread_Pool::join()
{
    _join = true;
    for (auto& t : _threads)
    {
        if (t.joinable()) t.join();
    }
}

const unsigned int& Thread_Pool::get_num_threads() const
{
    return _num_threads;
}

const std::vector<int>& Thread_Pool::get_core_numbers() const
{
    return _core_numbers;
}

void Thread_Pool::wait_until_queue_empty()
{
    while (_queue_size) std::this_thread::yield();
}

const bool Thread_Pool::queue_empty() const 
{ 
    return (_queue_size == 0); 
}

Thread_Pool::~Thread_Pool()
{
    join();
}

void Thread_Pool::thread_work(Thread_Pool* threadPool)
{
    std::function<void()> work;
    bool work_assigned = false;
    std::unique_lock<std::mutex> queue_lck(threadPool->_queue_mutex, std::defer_lock);
    while (!(threadPool->_join && threadPool->_work_queue.empty())) //break the loop if only join is called and queue is empty 
    {
        if (threadPool->_work_queue.empty()) std::this_thread::yield();
        else
        {
            queue_lck.lock();
            if (!threadPool->_work_queue.empty())
            {
                work = threadPool->_work_queue.front();
                threadPool->_work_queue.pop();
                work_assigned = true;
            }
            queue_lck.unlock();
            if (work_assigned)
            {
                work();
                threadPool->_queue_size--;
                work_assigned = false;
            }
        }
    }
}
#else
Thread_Pool::Thread_Pool(const unsigned int num_threads)
{
    _num_threads = (num_threads > 0) ? num_threads : std::thread::hardware_concurrency();
    for (unsigned int i = 0; i < _num_threads; i++)
    {
        _threads.emplace_back(std::thread(Thread_Pool::thread_work, this));
        const int core_no = Schedule_Thread(_threads[i]);
        _core_numbers.push_back(core_no);
    }
}

Thread_Pool::Thread_Pool(const Thread_Pool& thread_pool) :
    _num_threads(thread_pool.get_num_threads()),
    _core_numbers(thread_pool.get_core_numbers())
{
    for (unsigned int i = 0; i < _num_threads; i++)
    {
        _threads.emplace_back(std::thread(Thread_Pool::thread_work, this));
        const int core_no = Schedule_Thread(_threads[i], _core_numbers[i]);
        _core_numbers.push_back(core_no);
    }
}

void Thread_Pool::assign(std::function<void()> work)
{
    std::unique_lock<std::mutex> queue_lck(_queue_mutex);
    _work_queue.push(work);
    _queue_size++;
    queue_lck.unlock();
    _cv.notify_all();
}

void Thread_Pool::join()
{
    _join = true;
    _cv.notify_all();
    for (auto& t : _threads)
    {
        if (t.joinable()) t.join();
    }
}

void Thread_Pool::wait_until_queue_empty()
{
    while (_queue_size) std::this_thread::yield();
}

const bool Thread_Pool::queue_empty() const 
{ 
    return (_queue_size == 0);
}

const unsigned int& Thread_Pool::get_num_threads() const
{
    return _num_threads;
}

const std::vector<int>& Thread_Pool::get_core_numbers() const
{
    return _core_numbers;
}

Thread_Pool::~Thread_Pool()
{
    join();
}

void Thread_Pool::thread_work(Thread_Pool* threadPool)
{
    std::function<void()> work;
    bool work_assigned = false;
    std::unique_lock<std::mutex> cv_lck(threadPool->_cv_mutex, std::defer_lock);
    std::unique_lock<std::mutex> queue_lck(threadPool->_queue_mutex, std::defer_lock);
    //break the loop if only join is called and queue is empty 
    while (!(threadPool->_join && threadPool->_work_queue.empty()))
    {
        cv_lck.lock();
        //call wait only if join is not called and queue is empty 
        if (!threadPool->_join && threadPool->_work_queue.empty())
        {
            //wait only if join is not called and queue is empty
            threadPool->_cv.wait(cv_lck, [&threadPool]()
                {return !(!threadPool->_join && threadPool->_work_queue.empty()); });
        }
        cv_lck.unlock();

        queue_lck.lock();
        if (!threadPool->_work_queue.empty())
        {
            work = threadPool->_work_queue.front();
            threadPool->_work_queue.pop();
            work_assigned = true;
        }
        queue_lck.unlock();
        if (work_assigned)
        {
            work();
            threadPool->_queue_size--;
            work_assigned = false;
        }
    }
}
#endif