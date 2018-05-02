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
#include "infrastructure/io/input_deck.h"

#include <cassert>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm> // std::transform, std::fill

#include "common/error.h"
#include "common/sizes.h"
#include "utilities/misc/string_utils.h"
#include "packages/time/config.h"
#include "packages/hydro/config.h"
#include "packages/ale/config.h"
#include "utilities/data/global_configuration.h"
#include "packages/setup/mesh_region.h"
#include "utilities/eos/config.h"
#include "packages/setup/indicators.h"
#include "packages/setup/types.h"



namespace bookleaf {
namespace inf {
namespace io {

// Little helper for reading YAML nodes, expands to:
//
//  if (node["STR"])
//      BASE.STR = node["STR"].as<decltype(BASE.STR)>();
//
#define READ_NODE(BASE,STR) { \
    if (node[#STR]) BASE.STR = node[#STR].as<decltype(BASE.STR)>(); \
}

#define READ_BOOL_NODE(BASE,STR) { \
    if (node[#STR]) BASE.STR = (unsigned char) node[#STR].as<bool>(); \
}



void
InputDeck::open(std::string filename, Error &err)
{
    try {
        root = YAML::LoadFile(filename);
    } catch (...) {
        err.fail("ERROR: couldn't open YAML input deck " + filename);
    }
}



void
InputDeck::readTimeConfiguration(time::Config &time)
{
    if (root["TIME"]) {
        YAML::Node node = root["TIME"];
        READ_NODE(time, time_start);
        READ_NODE(time, time_end);
        READ_NODE(time, dt_initial);
        READ_NODE(time, dt_max);
        READ_NODE(time, dt_min);
        READ_NODE(time, dt_g);
    }
}



void
InputDeck::readHydroConfiguration(hydro::Config &hydro)
{
    if (root["HYDRO"]) {
        YAML::Node node = root["HYDRO"];
        READ_NODE(hydro, cvisc1);
        READ_NODE(hydro, cvisc2);
        READ_NODE(hydro, cfl_sf);
        READ_NODE(hydro, div_sf);
        READ_NODE(hydro, zhg);
        READ_NODE(hydro, zsp);
        READ_NODE(hydro, kappaall);
        READ_NODE(hydro, pmeritall);
        READ_NODE(hydro, kappareg);
        READ_NODE(hydro, pmeritreg);
        READ_NODE(hydro, zdtnotreg);
        READ_NODE(hydro, zmidlength);
    }
}



void
InputDeck::readALEConfiguration(ale::Config &ale)
{
    if (root["ALE"]) {
        YAML::Node node = root["ALE"];
        READ_NODE(ale, npatch);
        READ_NODE(ale, adv_type);
        READ_NODE(ale, mintime);
        READ_NODE(ale, maxtime);
        READ_NODE(ale, sf);
        READ_NODE(ale, zexist);
        READ_NODE(ale, zon);
        READ_BOOL_NODE(ale, zeul);
        READ_NODE(ale, patch_type);
        READ_NODE(ale, patch_motion);
        READ_NODE(ale, patch_ntrigger);
        READ_NODE(ale, patch_ontime);
        READ_NODE(ale, patch_offtime);
        READ_NODE(ale, patch_minvel);
        READ_NODE(ale, patch_maxvel);
        READ_NODE(ale, patch_om);
        READ_NODE(ale, patch_trigger);
    }
}



void
InputDeck::readMeshRegions(std::vector<setup::MeshRegion> &mesh_regions)
{
    if (root["MESH"]) {
        YAML::Node node = root["MESH"];
        for (auto child : node) {
            mesh_regions.push_back(readMeshRegion(child));
        }
    }
}



void
InputDeck::readGlobalConfiguration(GlobalConfiguration &gc)
{
    if (root["CUTOFF"]) {
        YAML::Node node = root["CUTOFF"];
        READ_NODE(gc, zcut);
        READ_NODE(gc, zerocut);
        READ_NODE(gc, dencut);
        READ_NODE(gc, accut);
    }
}



void
InputDeck::readEOS(EOS &eos)
{
    if (root["CUTOFF"]) {
        YAML::Node node = root["CUTOFF"];
        READ_NODE(eos, pcut);
        READ_NODE(eos, ccut);
    }

    if (root["EOS"]) {
        YAML::Node node = root["EOS"];
        for (auto child : node) {
            eos.mat_eos.push_back(readMaterialEOS(child));
        }
    }
}



void
InputDeck::readShapes(std::vector<setup::Shape> &shapes)
{
    using setup::Shape;

    if (root["SHAPES"]) {
        YAML::Node parent = root["SHAPES"];
        for (auto node : parent) {
            Shape shape;

            // Read shape type
            shape.type = Shape::Type::UNKNOWN;
            if (node["type"]) {
                std::string stype = to_upper(node["type"].as<std::string>());

                if      (stype == "RECTANGLE") shape.type = Shape::Type::RECTANGLE;
                else if (stype == "CIRCLE")    shape.type = Shape::Type::CIRCLE;
            }

            // Read shape parameters (ignore after NUM_PARAMS)
            std::fill(shape.params, shape.params + Shape::NUM_PARAMS, 0);
            if (node["params"]) {
                int i = 0;
                for (auto val : node["params"]) {
                    shape.params[i++] = val.as<double>();
                    if (i >= Shape::NUM_PARAMS) break;
                }
            }

            // Zero last parameter of circle
            if (shape.type == Shape::Type::CIRCLE) {
                shape.params[3] = 0.;
            }

            shapes.push_back(shape);
        }
    }
}



void
InputDeck::readIndicators(std::vector<setup::Region> &regions,
        std::vector<setup::Material> &materials)
{
    if (root["INDICATORS"]) {
        YAML::Node node = root["INDICATORS"];
        if (node["regions"]) {
            for (auto child : node["regions"]) {
                setup::Region region = readRegion(child);
                region.index = regions.size();
                regions.push_back(region);
            }
        }

        if (node["materials"]) {
            for (auto child : node["materials"]) {
                setup::Material material = readMaterial(child);
                material.index = materials.size();
                materials.push_back(material);
            }
        }
    }
}



void
InputDeck::readInitialConditions(std::vector<setup::ThermodynamicsIC> &thermo,
        std::vector<setup::KinematicsIC> &kinematics)
{
    if (root["INITIAL_CONDITIONS"]) {
        YAML::Node node = root["INITIAL_CONDITIONS"];

        if (node["thermodynamics"]) {
            for (auto child : node["thermodynamics"]) {
                thermo.push_back(readThermodynamicsInitialCondition(child));
            }
        }

        if (node["kinematics"]) {
            for (auto child : node["kinematics"]) {
                kinematics.push_back(readKinematicsInitialCondition(child));
            }
        }
    }
}



setup::MeshRegion
InputDeck::readMeshRegion(YAML::Node node)
{
    using setup::MeshRegion;

    auto read_mesh_region_side_segment =
        [](YAML::Node node) -> MeshRegion::Side::Segment {

        MeshRegion::Side::Segment mrss;

        // Read segment type
        if (node["type"]) {
            std::string stype = to_upper(node["type"].as<std::string>());

            if (stype == "LINE")
                mrss.type = MeshRegion::Side::Segment::Type::LINE;
            else if (stype == "ARC_C")
                mrss.type = MeshRegion::Side::Segment::Type::ARC_C;
            else if (stype == "ARC_A")
                mrss.type = MeshRegion::Side::Segment::Type::ARC_A;
            else if (stype == "POINT")
                mrss.type = MeshRegion::Side::Segment::Type::POINT;
            else if (stype == "LINK")
                mrss.type = MeshRegion::Side::Segment::Type::LINK;
        }

        // Read segment boundary condition
        if (node["bc"]) {
            std::string sbc = to_upper(node["bc"].as<std::string>());

            if (sbc == "SLIPX")
                mrss.bc = MeshRegion::Side::Segment::BoundaryCondition::SLIPX;
            else if (sbc == "SLIPY")
                mrss.bc = MeshRegion::Side::Segment::BoundaryCondition::SLIPY;
            else if (sbc == "WALL")
                mrss.bc = MeshRegion::Side::Segment::BoundaryCondition::WALL;
            else if (sbc == "TRANS")
                mrss.bc = MeshRegion::Side::Segment::BoundaryCondition::TRANS;
            else if (sbc == "OPEN")
                mrss.bc = MeshRegion::Side::Segment::BoundaryCondition::OPEN;
            else if (sbc == "FREE")
                mrss.bc = MeshRegion::Side::Segment::BoundaryCondition::FREE;
        }

        // Read segment positions
        if (node["pos"]) {
            std::vector<double> pos = node["pos"].as<std::vector<double>>();
            assert(pos.size() <= 8);
            std::copy(pos.begin(), pos.end(), mrss.pos);
        }

        return mrss;
    };

    auto read_mesh_region_side =
        [read_mesh_region_side_segment](YAML::Node node) -> MeshRegion::Side {

        // Each side contains a number of segments
        MeshRegion::Side mrs;
        for (auto child : node) {
            mrs.segments.push_back(read_mesh_region_side_segment(child));
        }

        return mrs;
    };

    MeshRegion mr;

    // Read region type
    if (node["type"]) {
        std::string stype = to_upper(node["type"].as<std::string>());

        if      (stype == "LIN1") mr.type = MeshRegion::Type::LIN1;
        else if (stype == "LIN2") mr.type = MeshRegion::Type::LIN2;
        else if (stype == "EQUI") mr.type = MeshRegion::Type::EQUI;
        else if (stype == "USER") mr.type = MeshRegion::Type::USER;
    }

    // Read region dimensions
    if (node["dims"]) {
        std::vector<double> dims = node["dims"].as<std::vector<double>>();
        std::copy(dims.begin(), dims.begin() + 2, mr.dims);

        // Fortran flips the dimensions for some reason
        std::swap(mr.dims[0], mr.dims[1]);
    }

    // Read region tol, om, material index
    READ_NODE(mr, tol);
    READ_NODE(mr, om);
    READ_NODE(mr, material);

    // Read region sides
    if (node["sides"]) {
        for (auto child : node["sides"]) {
            mr.sides.push_back(read_mesh_region_side(child));
        }
    }

    return mr;
}



MaterialEOS
InputDeck::readMaterialEOS(YAML::Node node)
{
    MaterialEOS meos;

    // Read EOS type
    if (node["type"]) {
        std::string stype = to_upper(node["type"].as<std::string>());

        if      (stype == "VOID")      meos.type = MaterialEOS::Type::VOID;
        else if (stype == "IDEAL GAS") meos.type = MaterialEOS::Type::IDEAL_GAS;
        else if (stype == "TAIT")      meos.type = MaterialEOS::Type::TAIT;
        else if (stype == "JWL")       meos.type = MaterialEOS::Type::JWL;
    }

    // Read params
    if (node["params"]) {
        std::vector<double> dims = node["params"].as<std::vector<double>>();
        std::copy(dims.begin(), dims.begin() + MaterialEOS::NUM_PARAMS,
                meos.params);
    }

    return meos;
}



setup::Region
InputDeck::readRegion(YAML::Node node)
{
    using setup::Region;

    Region region;

    // Read region type
    if (node["type"]) {
        std::string stype = to_upper(node["type"].as<std::string>());

        if      (stype == "MESH")
                    region.type = Region::Type::MESH;
        else if (stype == "MATERIAL")
                    region.type = Region::Type::MATERIAL;
        else if (stype == "BACKGROUND")
                    region.type = Region::Type::BACKGROUND;
        else if (stype == "CELL")
                    region.type = Region::Type::CELL;
        else if (stype == "SHAPE")
                    region.type = Region::Type::SHAPE;
    }

    // Read region value, name
    READ_NODE(region, value);
    READ_NODE(region, name);
    region.index = -1;

    return region;
}



setup::Material
InputDeck::readMaterial(YAML::Node node)
{
    using setup::Material;

    Material material;

    // Read material type
    if (node["type"]) {
        std::string stype = to_upper(node["type"].as<std::string>());

        if      (stype == "MESH")
                    material.type = Material::Type::MESH;
        else if (stype == "REGION")
                    material.type = Material::Type::REGION;
        else if (stype == "BACKGROUND")
                    material.type = Material::Type::BACKGROUND;
        else if (stype == "CELL")
                    material.type = Material::Type::CELL;
        else if (stype == "SHAPE")
                    material.type = Material::Type::SHAPE;
    }

    // Read material value, name
    READ_NODE(material, value);
    READ_NODE(material, name);
    material.index = -1;

    return material;
}



setup::ThermodynamicsIC
InputDeck::readThermodynamicsInitialCondition(YAML::Node node)
{
    using setup::ThermodynamicsIC;

    ThermodynamicsIC tic;

    if (node["type"]) {
        std::string stype = to_upper(node["type"].as<std::string>());

        if (stype == "REGION")
            tic.type = ThermodynamicsIC::Type::REGION;
        else if (stype == "MATERIAL")
            tic.type = ThermodynamicsIC::Type::MATERIAL;
    }

    READ_NODE(tic, value);

    if (node["energy_scale"]) {
        std::string sscale = to_upper(node["energy_scale"].as<std::string>());

        if (sscale == "VOLUME")
            tic.energy_scale = ThermodynamicsIC::EnergyScale::VOLUME;
        else if (sscale == "MASS")
            tic.energy_scale = ThermodynamicsIC::EnergyScale::MASS;
        else
            tic.energy_scale = ThermodynamicsIC::EnergyScale::ZERO;
    }

    READ_NODE(tic, density);
    READ_NODE(tic, energy);

    return tic;
}



setup::KinematicsIC
InputDeck::readKinematicsInitialCondition(YAML::Node node)
{
    using setup::KinematicsIC;

    KinematicsIC kic;

    // Read type
    if (node["type"]) {
        std::string stype = to_upper(node["type"].as<std::string>());

        if (stype == "BACKGROUND")
            kic.type = KinematicsIC::Type::BACKGROUND;
        else if (stype == "REGION")
            kic.type = KinematicsIC::Type::REGION;
    }

    // Read geometry
    if (node["geometry"]) {
        std::string sgeom = to_upper(node["geometry"].as<std::string>());

        if (sgeom == "RADIAL")
            kic.geometry = KinematicsIC::Geometry::RADIAL;
        else if (sgeom == "PLANAR")
            kic.geometry = KinematicsIC::Geometry::PLANAR;
    }

    // Read value
    READ_NODE(kic, value);

    // Read parameters (ignore after NUM_PARAMS)
    if (node["params"]) {
        int i = 0;
        for (auto val : node["params"]) {
            kic.params[i++] = val.as<double>();
            if (i >= KinematicsIC::NUM_PARAMS) break;
        }
    }

    // Zero last planar parameter
    if (kic.geometry == KinematicsIC::Geometry::PLANAR) {
        kic.params[2] = 0.;
    }

    return kic;
}

#undef READ_NODE

} // namespace io
} // namespace inf
} // namespace bookleaf
