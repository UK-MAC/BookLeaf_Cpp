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
#ifndef BOOKLEAF_COMMON_CONFIG_H
#define BOOKLEAF_COMMON_CONFIG_H

#include <memory>



namespace bookleaf {

namespace time { struct Config; }
namespace hydro { struct Config; }
namespace ale { struct Config; }
struct GlobalConfiguration;
struct EOS;
namespace geometry { struct Config; }
namespace io_utils { struct Labels; }
namespace comms { struct Comms; }
namespace setup { struct Config; }

struct Config
{
    std::shared_ptr<time::Config>        time;
    std::shared_ptr<hydro::Config>       hydro;
    std::shared_ptr<EOS>                 eos;
    std::shared_ptr<geometry::Config>    geom;
    std::shared_ptr<ale::Config>         ale;
    std::shared_ptr<io_utils::Labels>    io;
    std::shared_ptr<comms::Comms>        comms;
    std::shared_ptr<GlobalConfiguration> global;
    std::shared_ptr<setup::Config>       setup;

    Config();
};

} // namespace bookleaf



#endif // BOOKLEAF_COMMON_CONFIG_H
