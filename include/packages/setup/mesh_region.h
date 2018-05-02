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
#ifndef BOOKLEAF_PACKAGES_SETUP_MESH_REGION_H
#define BOOKLEAF_PACKAGES_SETUP_MESH_REGION_H

#include <vector>
#include <iostream>



namespace bookleaf {

struct Error;
struct Sizes;

namespace setup {

struct MeshRegion
{
    struct Side
    {
        struct Segment
        {
            // Integer values correspond to original Fortran code for
            // convenience when porting, have no significance other than that
            enum class Type : int {
                UNKNOWN=0, LINE=1, ARC_C=2, ARC_A=3, POINT=4, LINK=5 };
            enum class BoundaryCondition : int {
                SLIPX=1, SLIPY=2, WALL=3, TRANS=6, OPEN=7, FREE=8 };

            Type type = Type::UNKNOWN;
            BoundaryCondition bc = BoundaryCondition::FREE;
            double pos[8] = {0};
        };

        std::vector<Segment> segments;
    };

    enum class Type : int { UNKNOWN=0, LIN1=1, LIN2=2, EQUI=3, USER=4 };

    // Mesh region dimensions
    int dims[2] = {0};

    // Mesh region type
    Type type = Type::UNKNOWN;

    // Material index if specifying materials via mesh
    int material = -1;

    // Convergence tolerance/scaling factor
    double tol = 1.e-12;
    double om = 1.;

    // Number of iterations in for convergence
    int no_it = 0;

    // ???
    double *rr = nullptr;
    double *ss = nullptr;
    int *bc = nullptr;
    unsigned char *merged = nullptr;

    // Weights
    double r_wgt[8] = {0};
    double s_wgt[8] = {0};
    double wgt[16] = {0};

    std::vector<Side> sides;
};

void printMeshRegions(std::vector<MeshRegion> const &mesh_regions,
        Sizes const &sizes);

std::ostream &operator<<(std::ostream &os, MeshRegion const &rhs);

void rationaliseMeshRegions(std::vector<MeshRegion> &mesh_regions,
        Sizes const &sizes, Error &err);

#ifdef BOOKLEAF_DEBUG
void
dumpMeshRegionData(
        std::string filename,
        MeshRegion const &mesh_region);
#endif

} // namespace setup
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_SETUP_MESH_REGION_H
