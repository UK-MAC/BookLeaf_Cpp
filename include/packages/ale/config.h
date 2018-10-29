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
#ifndef BOOKLEAF_PACKAGES_ALE_CONFIG_H
#define BOOKLEAF_PACKAGES_ALE_CONFIG_H

#include <vector>
#include <limits>
#include <iostream>
#include <memory>

#include <cub/cub.cuh>

#include "common/defs.h"



namespace bookleaf {

struct Sizes;
struct Error;
struct GlobalConfiguration;
namespace comms { struct Comm; }
struct EOS;

namespace time { struct Config; }

namespace ale {

struct Config
{
    int npatch = 0;
    int adv_type = 1;
    double mintime = std::numeric_limits<double>::max();
    double maxtime = std::numeric_limits<double>::max();
    double sf = 0.5;
    unsigned char zexist = false;
    unsigned char zon = false;
    unsigned char zeul = false;

    std::vector<int> patch_type;
    std::vector<int> patch_motion;
    std::vector<int> patch_ntrigger;
    std::vector<double> patch_ontime;
    std::vector<double> patch_offtime;
    std::vector<double> patch_minvel;
    std::vector<double> patch_maxvel;
    std::vector<double> patch_om;
    std::vector<std::vector<int>> patch_trigger;

    std::shared_ptr<comms::Comm>         comm;
    std::shared_ptr<GlobalConfiguration> global;
    std::shared_ptr<EOS>                 eos;

    unsigned char *cub_storage = nullptr;
    SizeType cub_storage_len = 0;
    cub::KeyValuePair<int, double> *cub_out = nullptr;

    Config();
};

std::ostream &operator<<(std::ostream &os, ale::Config const &rhs);

void rationalise(ale::Config &ale, time::Config const &time, Error &err);

void
initALEConfig(
        Sizes const &sizes,
        ale::Config &ale,
        Error &err);

void
killALEConfig(
        ale::Config &ale);

} // namespace ale
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_ALE_CONFIG_H
