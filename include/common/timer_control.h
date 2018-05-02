/* @HEADER@
 * Crown Copyright 2018 AWE.
 *
 * This file is part of BookLeaf.
 *
 * BookLeaf is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 * 
 * BookLeaf is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * BookLeaf. If not, see http://www.gnu.org/licenses/.
 * @HEADER@ */
#ifndef BOOKLEAF_COMMON_TIMER_CONTROL_H
#define BOOKLEAF_COMMON_TIMER_CONTROL_H

#include <map>
#include <chrono>
#include <string>



namespace bookleaf {

struct Sizes;

typedef std::chrono::high_resolution_clock::time_point time_point;
typedef std::chrono::high_resolution_clock::duration duration;

enum class TimerID : int {
    TOTAL = 0,
    INIT,
    GETEOSI,
    GETGEOMETRYI,
    COMMREGISTER,
    MESHGEN,
    MESHPARTITION,
    SETUPIC,
    SOLVER,
    GETDT,
    GETVISCOSITY,
    COMMT,
    COLLECTIVET,
    LAGSTEP,
    GETEOSL,
    GETACCELERATION,
    GETGEOMETRYL,
    GETENERGY,
    GETFORCE,
    GETHG,
    GETSP,
    COMML,
    ALESTEP,
    ALEGETMESHSTATUS,
    ALEGETFLUXVOLUME,
    ALEADVECT,
    ALEADVECTEL,
    ALEADVECTBASISEL,
    ALEADVECTVAREL,
    ALEADVECTND,
    ALEADVECTBASISND,
    ALEADVECTVARND,
    ALEUPDATE,
    GETEOSA,
    GETGEOMETRYA,
    COMMA,
    STEPIO,
    IO,

    NTIMERS = IO + 1
};



struct Timer
{
    std::string tstring;
    TimerID id;
    time_point start;
    duration time;
    bool zactive;
};



class TimerControl
{
public:
    TimerControl();

    void start(Timer &t);
    void start(TimerID id);

    void stop(Timer &t);
    void stop(TimerID id);
    void stopAll();

    void add(TimerID id, std::string name);

    void reset(TimerID id);
    void reset(Timer &t);

    float getGrind(int nel, Timer &t);

    static float getCount(Timer const &t) {
        typedef std::chrono::duration<float> fsecs;
        return std::chrono::duration_cast<fsecs>(t.time).count();
    }

    friend std::ostream &operator<<(std::ostream &out,
            TimerControl const &timers);

private:
    std::map<TimerID, Timer> timers;

    static time_point getTime() {
        return std::chrono::high_resolution_clock::now();
    }

};



// Start a timer on construction and then stop it on destruction (RAII)
class ScopedTimer
{
public:
    explicit ScopedTimer(TimerControl &timers, TimerID id);
    ~ScopedTimer();

private:
    TimerControl &timers;
    TimerID id;
};

} // namespace bookleaf



#endif // BOOKLEAF_COMMON_TIMER_CONTROL_H
