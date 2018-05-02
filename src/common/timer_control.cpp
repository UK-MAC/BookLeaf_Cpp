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
#include "common/timer_control.h"

#include <iostream>
#include <iomanip>
#include <cassert>

#include "common/sizes.h"



namespace bookleaf {

TimerControl::TimerControl()
{
    // Creat a list of timers and add to the collection
    add(TimerID::TOTAL,            " total run time                    ");
    add(TimerID::INIT,             "   time in initialisation          ");
    add(TimerID::GETEOSI,          "     time in geteos                ");
    add(TimerID::GETGEOMETRYI,     "     time in getgeometry           ");
    add(TimerID::COMMREGISTER,     "     time in register              ");
    add(TimerID::MESHGEN,          "     time in mesh generator        ");
    add(TimerID::MESHPARTITION,    "     time in mesh partition        ");
    add(TimerID::SETUPIC,          "     time in initial conditions    ");
    add(TimerID::SOLVER,           "   time in solver                  ");
    add(TimerID::GETDT,            "     time in getdt                 ");
    add(TimerID::GETVISCOSITY,     "       time in getviscosity        ");
    add(TimerID::COMMT,            "         time in MPI exchanges     ");
    add(TimerID::COLLECTIVET,      "       time in MPI collectives     ");
    add(TimerID::LAGSTEP,          "     time in lagstep               ");
    add(TimerID::GETEOSL,          "       time in geteos              ");
    add(TimerID::GETACCELERATION,  "       time in getacceleration     ");
    add(TimerID::COMML,            "         time in MPI exchanges     ");
    add(TimerID::GETGEOMETRYL,     "       time in getgeometry         ");
    add(TimerID::GETENERGY,        "       time in getenergy           ");
    add(TimerID::GETFORCE,         "       time in getforce            ");
    add(TimerID::GETHG,            "       time in gethg               ");
    add(TimerID::GETSP,            "       time in getsp               ");
    add(TimerID::ALESTEP,          "     time in alestep               ");
    add(TimerID::ALEGETMESHSTATUS, "       time in getmeshstatus       ");
    add(TimerID::ALEGETFLUXVOLUME, "       time in getfluxvolume       ");
    add(TimerID::ALEADVECT,        "       time in advect              ");
    add(TimerID::ALEADVECTEL,      "         time in advectel          ");
    add(TimerID::ALEADVECTBASISEL, "           time in advectbasisel   ");
    add(TimerID::ALEADVECTVAREL,   "           time in advectvarel     ");
    add(TimerID::ALEADVECTND,      "         time in advectnd          ");
    add(TimerID::ALEADVECTBASISND, "           time in advectbasisnd   ");
    add(TimerID::ALEADVECTVARND,   "           time in advectvarnd     ");
    add(TimerID::COMMA,            "         time in MPI exchanges     ");
    add(TimerID::ALEUPDATE,        "       time in update              ");
    add(TimerID::GETEOSA,          "         time in geteos            ");
    add(TimerID::GETGEOMETRYA,     "         time in getgeometry       ");
    add(TimerID::STEPIO,           "     time in step IO               ");
    add(TimerID::IO,               "     time in output dumps          ");
}




void
TimerControl::start(Timer &t)
{
    if (!t.zactive) {
        t.start = getTime();
        t.zactive = true;
    }
}

void
TimerControl::start(TimerID id)
{
    auto it = timers.find(id);
    if (it != timers.end()) {
        Timer &t = it->second;
        start(t);

    } else {
        assert(false && "timer not found");
    }
}



void
TimerControl::stop(Timer &t)
{
    if (t.zactive) {
        t.time += getTime() - t.start;
        t.zactive = false;
    }
}

void
TimerControl::stop(TimerID id)
{
    auto it = timers.find(id);
    if (it != timers.end()) {
        Timer &t = it->second;
        stop(t);

    } else {
        assert(false && "timer not found");
    }
}

void
TimerControl::stopAll()
{
    for (auto it : timers) stop(it.second);
}



void
TimerControl::add(TimerID id, std::string name)
{
    Timer t;
    t.id      = id;
    t.tstring = name;
    t.start   = getTime();
    t.time    = duration(0);
    t.zactive = false;

    this->timers.insert(std::make_pair(id, t));
}



void
TimerControl::reset(TimerID id)
{
    auto it = timers.find(id);
    if (it != timers.end()) {
        Timer &t = it->second;
        reset(t);

    } else {
        assert(false && "timer not found");
    }
}



void
TimerControl::reset(Timer &t)
{
    t.start   = getTime();
    t.time    = duration(0);
    t.zactive = false;
}



float
TimerControl::getGrind(int nel, Timer &t)
{
    typedef std::chrono::duration<float> fsecs;
    float const secs = std::chrono::duration_cast<fsecs>(t.time).count();
    return secs * 1.e6 / nel;
}



std::ostream &
operator<<(std::ostream &os, TimerControl const &rhs)
{
    float secs = rhs.getCount(rhs.timers.at(TimerID::TOTAL));
    float const fac = 100.f / secs;

    for (auto it = rhs.timers.begin(); it != rhs.timers.end(); ++it) {

        secs = rhs.getCount(it->second);
        if (secs <= 0.f) continue;

        os << std::setw(35) << it->second.tstring;
        os << " ";
        os << std::scientific << std::setprecision(6) << std::setw(13) << secs;
        os << " ";
        os << std::fixed << std::setprecision(3) << std::setw(7) << secs * fac;
        os << " %";
        os << "\n";
    }

    return os;
}



ScopedTimer::ScopedTimer(TimerControl &timers, TimerID id) :
    timers(timers),
    id(id)
{
    timers.start(id);
}



ScopedTimer::~ScopedTimer()
{
    timers.stop(id);
}

} // namespace bookleaf
