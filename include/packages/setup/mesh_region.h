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
#include <cassert>



namespace bookleaf {

struct Error;
struct Sizes;

namespace setup {

/**
 * \author timrlaw
 *
 * Wraps the data specifying a mesh, which is subsequently used for generation.
 */
class MeshDescriptor
{
public:
    // Segment type
    enum class SegmentType : int {
        UNKNOWN = 0,
        LINE    = 1,
        ARC_C   = 2,
        ARC_A   = 3,
        POINT   = 4,
        LINK    = 5
    };

    // Segment boundary condition
    enum class SegmentBC : int {
        SLIPX = 1,
        SLIPY = 2,
        WALL  = 3,
        TRANS = 6,
        OPEN  = 7,
        FREE  = 8
    };

    // A mesh segment
    struct Segment
    {
        SegmentType type = SegmentType::UNKNOWN;
        SegmentBC   bc   = SegmentBC::FREE;

        // Data specifying segment (interpreted depending on type)
        double pos[8] = {0};
    };

    // A mesh side (collection of segments)
    typedef std::vector<Segment> Side;

    // Specify the type of mesh
    enum class Type : int {
        UNKNOWN = 0,
        LIN1    = 1,
        LIN2    = 2,
        EQUI    = 3,
        USER    = 4
    };

    // Mesh region dimensions (elements)
    int dims[2] = {0};

    // Mesh region type
    Type type = Type::UNKNOWN;

    // Material index if specifying materials via mesh
    int material = -1;

    // Convergence tolerance
    double tol = 1.e-12;

    // Convergence scaling factor
    double om = 1.;

    // Mesh sides
    std::vector<Side> sides;
};



/**
 * \author timrlaw
 *
 * Wraps the data representing a generated mesh, which is subsequently copied
 * into the main data arrays.
 */
class MeshData
{
public:
    MeshData(int dimx, int dimy);
    ~MeshData();

    // Mesh region dimensions (elements)
    int dims[2] = {0};

    // Has the data been allocated?
    bool allocated = false;

    // Number of iterations to convergence
    int no_it = 0;

    // Node positions
    double *ss = nullptr;   // ndx
    double *rr = nullptr;   // ndy

    // Boundary conditions
    int *bc = nullptr;

    // Numbering and ordering for elements and nodes
    int *nn = nullptr;
    int *en = nullptr;
    int *no  = nullptr;
    int *eo  = nullptr;

    double r_wgt[8]  = {0};
    double s_wgt[8]  = {0};
    double   wgt[16] = {0};

    // Allocate memory based on dims
    void allocate();

    // Deallocate memory
    void deallocate();

    int
    getNodeOrdering() const
    {
        #define IX(i, j) ((j) * (dims[0]+1) + (i))

        double const r1 = rr[IX(1, 0)] - rr[IX(0, 0)];
        double const r2 = rr[IX(0, 1)] - rr[IX(0, 0)];
        double const r3 = rr[IX(1, 1)] - rr[IX(0, 0)];
        double const s1 = ss[IX(1, 0)] - ss[IX(0, 0)];
        double const s2 = ss[IX(0, 1)] - ss[IX(0, 0)];
        double const s3 = ss[IX(1, 1)] - ss[IX(0, 0)];

        bool const cond =
            ((r1*s2-r2*s1) > 0.) ||
            ((r1*s3-r3*s1) > 0.) ||
            ((r3*s2-r2*s3) > 0.);

        return cond ? 1 : 0;

        #undef IX
    }
};



void
rationaliseMeshDescriptor(
        MeshDescriptor &md,
        Sizes const &sizes,
        Error &err);

void
printMeshDescriptor(
        MeshDescriptor const &mdesc,
        Sizes const &sizes);

#ifdef BOOKLEAF_DEBUG
void
dumpMeshData(
        std::string filename,
        MeshData const &mdata);
#endif

} // namespace setup
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_SETUP_MESH_REGION_H
