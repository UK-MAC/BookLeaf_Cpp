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
#ifndef BOOKLEAF_INFRASTRUCTURE_IO_INPUT_DECK_H
#define BOOKLEAF_INFRASTRUCTURE_IO_INPUT_DECK_H

#include <yaml-cpp/yaml.h>
#include <vector>
#include <string>



namespace bookleaf {

struct Error;
namespace time { struct Config; }
namespace hydro { struct Config; }
namespace ale { struct Config; }
struct GlobalConfiguration;

namespace setup {
struct Region;
struct Material;
struct MeshDescriptor;
struct Shape;
struct ThermodynamicsIC;
struct KinematicsIC;
}

struct EOS;
struct MaterialEOS;

namespace inf {
namespace io {

class InputDeck
{
public:
    void open(std::string input_file, Error &err);

    void readTimeConfiguration(time::Config &time, Error &err);
    void readHydroConfiguration(hydro::Config &hydro, Error &err);
    void readALEConfiguration(ale::Config &ale, Error &err);
    void readGlobalConfiguration(GlobalConfiguration &gc, Error &err);
    void readMesh(setup::MeshDescriptor &md);
    void readEOS(EOS &eos, Error &err);
    void readShapes(std::vector<setup::Shape> &shapes);
    void readIndicators(std::vector<setup::Region> &regions,
            std::vector<setup::Material> &materials);
    void readInitialConditions(std::vector<setup::ThermodynamicsIC> &thermo,
            std::vector<setup::KinematicsIC> &kinematics);

private:
    YAML::Node root;

    static MaterialEOS readMaterialEOS(YAML::Node node, Error &err);
    static setup::Region readRegion(YAML::Node node);
    static setup::Material readMaterial(YAML::Node node);
    static setup::ThermodynamicsIC readThermodynamicsInitialCondition(YAML::Node node);
    static setup::KinematicsIC readKinematicsInitialCondition(YAML::Node node);
};

} // namespace io
} // namespace inf
} // namespace bookleaf



#endif // BOOKLEAF_INFRASTRUCTURE_IO_INPUT_DECK_H
