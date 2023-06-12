#pragma once
#include <thread>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <queue>
#include <iostream>

#if __linux__ == 0
#include <Windows.h>
#endif

#define THREAD_POOL_CV 1

#if __linux__ == 1
void Reset_Core_Counter();
void Check_Multi_Thread_Compatibility();
const int Schedule_Thread(std::thread& thread, const int core_number = -1);
void Print_Thread_Info();
#else
void Reset_Core_Counter();
void Check_Multi_Thread_Compatibility();
const int Schedule_Thread(std::thread& thread, const int core_number = -1);
void Print_Thread_Info();
#endif

#if THREAD_POOL_CV == 0
class Thread_Pool
{
public:
    Thread_Pool(const unsigned int num_threads = 0);
    Thread_Pool(const Thread_Pool& thread_pool);
    void assign(std::function<void()> work);
    void join();
    const unsigned int& get_num_threads() const;
    const std::vector<int>& get_core_numbers() const;
    void wait_until_queue_empty();
    const bool queue_empty() const;
    ~Thread_Pool();
private:
    static void thread_work(Thread_Pool* threadPool);

    std::vector<int> _core_numbers{};
    std::atomic_bool _join{ false };
    std::mutex _queue_mutex;
    std::queue<std::function<void()>> _work_queue{};
    unsigned int _num_threads;
    std::vector<std::thread> _threads;
    std::atomic_uint16_t _queue_size{ 0 };
};
#else
class Thread_Pool
{
public:
    Thread_Pool(const unsigned int num_threads = 0);
    Thread_Pool(const Thread_Pool& thread_pool);
    void assign(std::function<void()> work);
    void join();
    void wait_until_queue_empty();
    const bool queue_empty() const;
    const unsigned int& get_num_threads() const;
    const std::vector<int>& get_core_numbers() const;
    ~Thread_Pool();
private:
    static void thread_work(Thread_Pool* threadPool);
    
    std::vector<int> _core_numbers{};
    std::vector<std::thread> _threads;
    std::mutex _queue_mutex, _cv_mutex;
    std::condition_variable _cv;
    std::atomic_bool _join{ false };
    unsigned int _num_threads{ 0 };
    std::atomic_uint16_t _queue_size{ 0 };
    std::queue<std::function<void()>> _work_queue{};
};
#endif