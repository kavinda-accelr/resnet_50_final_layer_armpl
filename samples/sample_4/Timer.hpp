#pragma once
#include <string>
#include <map>
#include <iostream>
#include <iomanip>

#if __linux__ == 1
struct Time_data
{
    double t_cpu_time_used = 0;
    double t_process_cpu = 0;
    double t_monotonic = 0;
    double t_monotonic_raw = 0;
    double t_real = 0;
    unsigned int cycles = 0;
};
#else
struct Time_data
{
    double time = 0.0;
    unsigned int cycles = 0;
};
#endif
class Base_Timer
{
public:
    Base_Timer() {}
    virtual void start(std::string map_name) = 0;
    virtual void stop() = 0;
    virtual void reset() = 0;
    virtual void print_duration(bool header) = 0;
    virtual ~Base_Timer() {}
protected:
    std::string m_map_name;
    std::map<std::string, Time_data> m_time_data;
};
#if __linux__ == 1
#include <ctime>
class Timer : private Base_Timer
{
public:
    Timer(const Timer&) = delete;
    static Timer& Get();
    void start(std::string map_name) override;
    void stop() override;
    void reset() override;
    void print_duration(bool header=true) override;
    ~Timer();
private:
    Timer();
    struct timespec t_process_cpu_1, t_monotonic_1, t_monotonic_raw_1, t_real_1;
    struct timespec t_process_cpu_2, t_monotonic_2, t_monotonic_raw_2, t_real_2;
    clock_t t1, t2;
};
#else
#include <chrono>
class Timer : private Base_Timer
{
public:
    Timer(const Timer&) = delete;
    static Timer& Get();
    void start(std::string map_name) override;
    void stop() override;
    void reset() override;
    void print_duration(bool header=true) override;
    ~Timer();
private:
    Timer();
    std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimePoint;
};
#endif