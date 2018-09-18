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
#include "packages/setup/mesh_region.h"

#include <cassert>
#include <algorithm> // std::count_if
#include <fstream>
#include <bitset>

#ifdef BOOKLEAF_MPI_SUPPORT
#include <typhon.h>
#endif

#include "common/error.h"
#include "common/sizes.h"
#include "infrastructure/io/output_formatting.h"
#include "common/data_control.h"



namespace bookleaf {
namespace setup {

MeshData::MeshData(
        int dimx,
        int dimy)
    :
        dims { dimx, dimy }
{
    allocate();
}



MeshData::~MeshData()
{
    deallocate();
}



void
MeshData::allocate()
{
    if (allocated) {
        deallocate();
    }

    int const nel = dims[0] * dims[1];
    int const nnd = (dims[0]+1) * (dims[1]+1);

    ss = new double[nnd];
    rr = new double[nnd];
    bc = new int[nnd];
    nn = new int[nnd];
    en = new int[nel];
    no = new int[nnd];
    eo = new int[nel];

    std::fill(ss, ss + nnd, 0.0);
    std::fill(rr, rr + nnd, 0.0);
    std::fill(bc, bc + nnd, 0);
    std::fill(nn, nn + nnd, 0);
    std::fill(en, en + nel, 0);
    std::fill(no, no + nnd, 0);
    std::fill(eo, eo + nel, 0);

    allocated = true;
}



void
MeshData::deallocate()
{
    if (allocated) {
        delete[] eo;
        delete[] no;
        delete[] en;
        delete[] nn;
        delete[] bc;
        delete[] rr;
        delete[] ss;

        eo = nullptr;
        no = nullptr;
        en = nullptr;
        nn = nullptr;
        bc = nullptr;
        rr = nullptr;
        ss = nullptr;

        allocated = false;
    }
}



void
rationaliseMeshDescriptor(
        MeshDescriptor &md,
        Sizes const &sizes,
        Error &err)
{
    // Check whether specifying material via mesh
    bool const zflag = md.material != -1;

    auto &sides = md.sides;

    if (md.type == MeshDescriptor::Type::UNKNOWN) {
        err.fail("ERROR: unrecognised mesh type");
        return;
    }

    if (md.dims[0] <= 0) {
        err.fail("ERROR: mesh L dimension <= 0.0");
        return;
    }

    if (md.dims[1] <= 0) {
        err.fail("ERROR: mesh K dimension <= 0.0");
        return;
    }

    if (md.tol < 0.0) {
        err.fail("ERROR: mesh tol < 0.0");
        return;
    }

    if (md.om < 0.0) {
        err.fail("ERROR: mesh om < 0.0");
        return;
    }

    if (zflag && (md.material < 0 || md.material >= sizes.nmat)) {
        err.fail("ERROR: mesh material index out of range");
        return;
    }

    for (int j = 0; j < (int) sides.size(); j++) {
        MeshDescriptor::Side &segments = sides[j];

        for (int k = 0; k < (int) segments.size(); k++) {
            MeshDescriptor::Segment &seg = segments[k];
            if (seg.type == MeshDescriptor::SegmentType::UNKNOWN) {
                err.fail("ERROR: unrecognised segment type");
                return;
            }

            // Default boundary condition type is FREE so don't need to check
        }

        // If this is a multi-segment side...
        if (segments.size() > 1) {
            auto is_point = [](MeshDescriptor::Segment const &seg) {
                return seg.type == MeshDescriptor::SegmentType::POINT;
            };

            if (std::any_of(segments.begin(), segments.end(), is_point)) {
                err.fail("ERROR: point as part of multi-segment side");
                return;
            }

            // Check if the first two segments are connected by trying all
            // the permutations of their start and end points
            double l1, k1, l2, k2, l3, k3, l4, k4;

            l1 = segments[0].pos[2];
            k1 = segments[0].pos[3];
            l2 = segments[1].pos[0];
            k2 = segments[1].pos[1];

            if ((l1 != l2) || (k1 != k2)) {
                l3 = segments[1].pos[2];
                k3 = segments[1].pos[3];

                if ((l1 != l3) || (k1 != k2)) {
                    l4 = segments[0].pos[0];
                    k4 = segments[0].pos[1];

                    if ((l4 != l2) || (k4 != k2)) {
                        if ((l4 != l3) || (k4 != k3)) {
                            err.fail("ERROR: segment failed to form "
                                     "continuous side");
                            return;
                        } else {
                            segments[0].pos[0] = l1;
                            segments[0].pos[1] = k1;
                            segments[0].pos[2] = l4;
                            segments[0].pos[3] = k4;
                            segments[1].pos[0] = l3;
                            segments[1].pos[1] = k3;
                            segments[1].pos[2] = l2;
                            segments[1].pos[3] = k2;
                        }
                    } else {
                        segments[0].pos[0] = l1;
                        segments[0].pos[1] = k1;
                        segments[0].pos[2] = l4;
                        segments[0].pos[3] = k4;
                    }
                } else {
                    segments[1].pos[0] = l3;
                    segments[1].pos[1] = k3;
                    segments[1].pos[2] = l2;
                    segments[1].pos[3] = k2;
                }
            }

            // If there's more than two segments in the side then use the
            // corrected first two segments to correct the rest.
            if (segments.size() > 2) {
                for (int k = 1; k < (int) segments.size() - 1; k++) {
                    l1 = segments[k].pos[2];
                    k1 = segments[k].pos[3];
                    l2 = segments[k+1].pos[0];
                    k2 = segments[k+1].pos[1];

                    if ((l1 != l2) || (k1 != k2)) {
                        l3 = segments[k+1].pos[2];
                        k3 = segments[k+1].pos[3];

                        if ((l1 != l3) || (k1 != k3)) {
                            err.fail("ERROR: segment failed to form "
                                     "continuous side");
                            return;
                        } else {
                            segments[k+1].pos[0] = l3;
                            segments[k+1].pos[1] = k3;
                            segments[k+1].pos[2] = l2;
                            segments[k+1].pos[3] = k2;
                        }
                    }
                }
            } // segments.size() > 2
        } // segments.size() > 1
    } // for each side

    // Check the sides form a closed polygon
    assert(sides.size() == 4);
    double l1, k1, l2, k2, l3, k3, l4, k4;
    int nsegs1, nsegs2;

    nsegs1 = sides[0].size();
    l1 = sides[0][nsegs1-1].pos[2];
    k1 = sides[0][nsegs1-1].pos[3];
    l2 = sides[1][0].pos[0];
    k2 = sides[1][0].pos[1];

    if ((l1 != l2) || (k1 != k2)) {
        nsegs2 = sides[1].size();
        l3 = sides[1][nsegs2-1].pos[2];
        k3 = sides[1][nsegs2-1].pos[3];

        if ((l1 != l3) || (k1 != k3)) {
            l4 = sides[0][0].pos[0];
            k4 = sides[0][0].pos[1];

            if ((l4 != l2) || (k4 != k2)) {
                if ((l4 != l3) || (k4 != k3)) {
                    err.fail("ERROR: mesh region not closed");
                    return;
                } else {
                    if (nsegs1 == 1) {
                        sides[0][0].pos[0] = l1;
                        sides[0][0].pos[1] = k1;
                        sides[0][0].pos[2] = l4;
                        sides[0][0].pos[3] = k4;
                    } else {
                        for (int l = 0; l < nsegs1; l++) {
                            std::swap(
                                sides[0][l].pos[0],
                                sides[0][l].pos[2]);
                            std::swap(
                                sides[0][l].pos[1],
                                sides[0][l].pos[3]);
                        }
                    }

                    if (nsegs2 == 1) {
                        sides[1][0].pos[0] = l3;
                        sides[1][0].pos[1] = k3;
                        sides[1][0].pos[2] = l2;
                        sides[1][0].pos[3] = k2;
                    } else {
                        for (int l = 0; l < nsegs2; l++) {
                            std::swap(
                                sides[1][l].pos[0],
                                sides[1][l].pos[2]);
                            std::swap(
                                sides[1][l].pos[1],
                                sides[1][l].pos[3]);
                        }
                    }
                }
            } else {
                if (nsegs1 == 1) {
                    sides[0][0].pos[0] = l1;
                    sides[0][0].pos[1] = k1;
                    sides[0][0].pos[2] = l4;
                    sides[0][0].pos[3] = k4;
                } else {
                    for (int l = 0; l < nsegs1; l++) {
                        std::swap(
                            sides[0][l].pos[0],
                            sides[0][l].pos[2]);
                        std::swap(
                            sides[0][l].pos[1],
                            sides[0][l].pos[3]);
                    }
                }
            }
        } else {
            if (nsegs2 == 1) {
                sides[1][0].pos[0] = l3;
                sides[1][0].pos[1] = k3;
                sides[1][0].pos[2] = l2;
                sides[1][0].pos[3] = k2;
            } else {
                for (int l = 0; l < nsegs2; l++) {
                    std::swap(
                        sides[1][l].pos[0],
                        sides[1][l].pos[2]);
                    std::swap(
                        sides[1][l].pos[1],
                        sides[1][l].pos[3]);
                }
            }
        }
    }

    // Correct the remaining sides
    for (int k = 1; k < 3; k++) {
        nsegs1 = sides[k].size();
        l1 = sides[k][nsegs1-1].pos[2];
        k1 = sides[k][nsegs1-1].pos[3];
        l2 = sides[k+1][0].pos[0];
        k2 = sides[k+1][0].pos[1];

        if ((l1 != l2) || (k1 != k2)) {
            nsegs2 = sides[k+1].size();
            l3 = sides[k+1][nsegs2-1].pos[2];
            k3 = sides[k+1][nsegs2-1].pos[3];

            if ((l1 != l3) || (k1 != k3)) {
                err.fail("ERROR: region not closed");
                return;
            } else {
                if (nsegs2 == 1) {
                    sides[k+1][0].pos[0] = l3;
                    sides[k+1][0].pos[1] = k3;
                    sides[k+1][0].pos[2] = l2;
                    sides[k+1][0].pos[3] = k2;
                } else {
                    for (int l = 0; l < nsegs2; l++) {
                        std::swap(
                            sides[k+1][l].pos[0],
                            sides[k+1][l].pos[2]);
                        std::swap(
                            sides[k+1][l].pos[1],
                            sides[k+1][l].pos[3]);
                    }
                }
            }
        }
    }

    nsegs1 = sides[3].size();
    l1 = sides[3][nsegs1-1].pos[2];
    k1 = sides[3][nsegs1-1].pos[3];
    l2 = sides[0][0].pos[0];
    k2 = sides[0][0].pos[1];

    if ((l1 != l2) || (k1 != k2)) {
        err.fail("ERROR: mesh not closed");
        return;
    }
}



void
printMeshDescriptor(
        MeshDescriptor const &mdesc,
        Sizes const &sizes)
{
    int const num_mesh_materials = (mdesc.material != -1) ? 1 : 0;

    // Mesh materials
    std::cout <<
        inf::io::format_value("Number of mesh materials", "nmat",
            num_mesh_materials);

    // Number of elements and nodes
    std::cout <<
        inf::io::format_value("Number of elements", "nel", sizes.nel);
    std::cout <<
        inf::io::format_value("Number of nodes", "nnd", sizes.nnd);

    // Material index
    std::cout <<
        inf::io::format_value("Material index", "mesh_mat", mdesc.material);

    // Mesh type
    std::string mtype;
    switch (mdesc.type) {
    case MeshDescriptor::Type::LIN1:
        mtype = "LIN1"; break;
    case MeshDescriptor::Type::LIN2:
        mtype = "LIN2"; break;
    case MeshDescriptor::Type::EQUI:
        mtype = "EQUI"; break;
    case MeshDescriptor::Type::USER:
        mtype = "USER"; break;
    default:
        break;
    }

    std::cout << inf::io::format_value("Mesh type", "mesh_typ", mtype);

    // Dimensions
    std::cout <<
        inf::io::format_value("Mesh dimensions", "mesh_dims",
            std::to_string(mdesc.dims[0]) + "x" + std::to_string(mdesc.dims[1]));

    // Generation parameters
    std::cout <<
        inf::io::format_value("Convergence tolerance", "mesh_tol", mdesc.tol);
    std::cout <<
        inf::io::format_value("Convergence scaling factor", "mesh_om", mdesc.om);

    // Mesh sides
    for (int j = 0; j < (int) mdesc.sides.size(); j++) {
        MeshDescriptor::Side const &side = mdesc.sides[j];

        std::cout << "   Side: " << j << "\n";
        std::cout <<
            inf::io::format_value("  Number of segments", "", side.size());

        // Side segments
        for (int k = 0; k < (int) side.size(); k++) {
            MeshDescriptor::Segment const &seg = side[k];

            std::cout << "    Segment: " << k << "\n";

            // Segment type
            std::string type;
            switch (seg.type) {
            case MeshDescriptor::SegmentType::LINE:
                type = "LINE";  break;
            case MeshDescriptor::SegmentType::ARC_C:
                type = "ARC_C"; break;
            case MeshDescriptor::SegmentType::ARC_A:
                type = "ARC_A"; break;
            case MeshDescriptor::SegmentType::POINT:
                type = "POINT"; break;
            case MeshDescriptor::SegmentType::LINK:
                type = "LINK"; break;
            default:
                type = ""; break;
            }

            std::cout << inf::io::format_value("   Segment type", "", type);

            // Segment pos
            std::string pos;
            switch (seg.type) {
            case MeshDescriptor::SegmentType::LINE:
                pos = "(" + std::to_string(seg.pos[0]) + ", " +
                            std::to_string(seg.pos[1]) + ") - " +
                      "(" + std::to_string(seg.pos[2]) + ", " +
                            std::to_string(seg.pos[3]) + ")";

                std::cout <<
                    inf::io::format_value("   Segment position", "", pos);
                break;

            case MeshDescriptor::SegmentType::ARC_C:
                pos = "(" + std::to_string(seg.pos[0]) + ", " +
                            std::to_string(seg.pos[1]) + ") - " +
                      "(" + std::to_string(seg.pos[2]) + ", " +
                            std::to_string(seg.pos[3]) + ")";

                std::cout <<
                    inf::io::format_value("   Segment position", "", pos);

                pos = "about (" + std::to_string(seg.pos[4]) + ", " +
                                  std::to_string(seg.pos[5]) + ")";

                std::cout <<
                    inf::io::format_value("", "", pos);
                break;

            case MeshDescriptor::SegmentType::ARC_A:
                pos = "(" + std::to_string(seg.pos[0]) + ", " +
                            std::to_string(seg.pos[1]) + ") - " +
                      "(" + std::to_string(seg.pos[2]) + ", " +
                            std::to_string(seg.pos[3]) + ")";

                std::cout <<
                    inf::io::format_value("   Segment position", "", pos);

                pos = "about (" + std::to_string(seg.pos[4]) + ", " +
                                  std::to_string(seg.pos[5]) + ")";

                std::cout <<
                    inf::io::format_value("", "", pos);
                break;

            case MeshDescriptor::SegmentType::POINT:
                pos = "(" + std::to_string(seg.pos[0]) + ", " +
                            std::to_string(seg.pos[1]) + ")";

                std::cout <<
                    inf::io::format_value("   Segment position", "", pos);
                break;

            default:
                break;
            }

            // Segment boundary condition
            std::string bc;
            switch (seg.bc) {
            case MeshDescriptor::SegmentBC::SLIPX:
                bc = "SLIPX"; break;
            case MeshDescriptor::SegmentBC::SLIPY:
                bc = "SLIPY"; break;
            case MeshDescriptor::SegmentBC::WALL:
                bc = "WALL"; break;
            case MeshDescriptor::SegmentBC::TRANS:
                bc = "TRANS"; break;
            case MeshDescriptor::SegmentBC::OPEN:
                bc = "OPEN"; break;
            case MeshDescriptor::SegmentBC::FREE:
                bc = "FREE"; break;
            }

            std::cout <<
                inf::io::format_value("   Segment boundary condition", "", bc);
        }
    }
}



#ifdef BOOKLEAF_DEBUG
void
dumpMeshData(
        std::string filename,
        MeshData const &mdata)
{
    Error err;

#ifdef BOOKLEAF_MPI_SUPPORT
    // Serialise the dumps per-process if we're running under MPI
    int nproc, rank;
    TYPH_Get_Size(&nproc);
    TYPH_Get_Rank(&rank);

    filename += std::to_string(rank);

    TYPH_Barrier();
    for (int ip = 0; ip < nproc; ip++) {
        if (ip == rank) {
#endif

            std::ofstream of(filename.c_str());
            if (!of.is_open()) {
                FAIL_WITH_LINE(err, "ERROR: couldn't open file " + filename);
                std::exit(EXIT_FAILURE);
            }

            //unsigned long long *ptr;
            //std::bitset<64> bs;

            //int const no_l = mdata.dims[0] + 1;
            //int const no_k = mdata.dims[1] + 1;
            //#define IXm(i, j) (index2D(i, j, no_l))

            //// Dims
            //of << "dims\n";
            //of << 2 << "\n";
            //of << "integer" << "\n";
            //of << mr.dims[0] << "\n";
            //of << mr.dims[1] << "\n";
            //of << "\n";

            //// Type
            //of << "type\n";
            //of << 1 << "\n";
            //of << "integer" << "\n";
            //of << (int) mr.type << "\n";
            //of << "\n";

            //// Material
            //of << "material\n";
            //of << 1 << "\n";
            //of << "integer" << "\n";
            //of << mr.material << "\n";
            //of << "\n";

            //// Convergence/tolerance
            //of << "tol_om\n";
            //of << 2 << "\n";
            //of << "double" << "\n";
            //ptr = (unsigned long long *) &mr.tol;
            //bs = std::bitset<64>(*ptr);
            //of << bs << "\n";
            //ptr = (unsigned long long *) &mr.om;
            //bs = std::bitset<64>(*ptr);
            //of << bs << "\n";
            //of << "\n";

            //// # iterations
            //of << "no_it\n";
            //of << 1 << "\n";
            //of << "integer" << "\n";
            //of << mr.no_it << "\n";
            //of << "\n";

            //// rr
            //of << "rr\n";
            //of << no_l * no_k << "\n";
            //of << "double" << "\n";
            //for (int kk = 0; kk < no_k; kk++) {
                //for (int ll = 0; ll < no_l; ll++) {
                    //ptr = (unsigned long long *) &mr.rr[IXm(ll, kk)];
                    //bs = std::bitset<64>(*ptr);
                    //of << bs << "\n";
                //}
            //}
            //of << "\n";

            //// ss
            //of << "ss\n";
            //of << no_l * no_k << "\n";
            //of << "double" << "\n";
            //for (int kk = 0; kk < no_k; kk++) {
                //for (int ll = 0; ll < no_l; ll++) {
                    //ptr = (unsigned long long *) &mr.ss[IXm(ll, kk)];
                    //bs = std::bitset<64>(*ptr);
                    //of << bs << "\n";
                //}
            //}
            //of << "\n";

            //// bc
            //of << "bc\n";
            //of << no_l * no_k << "\n";
            //of << "integer" << "\n";
            //for (int kk = 0; kk < no_k; kk++) {
                //for (int ll = 0; ll < no_l; ll++) {
                    //of << mr.bc[IXm(ll, kk)] << "\n";
                //}
            //}
            //of << "\n";

            //// r_wgt
            //of << "r_wgt\n";
            //of << 8 << "\n";
            //of << "double" << "\n";
            //for (int i = 0; i < 8; i++) {
                //ptr = (unsigned long long *) &mr.r_wgt[i];
                //bs = std::bitset<64>(*ptr);
                //of << bs << "\n";
            //}
            //of << "\n";

            //// s_wgt
            //of << "s_wgt\n";
            //of << 8 << "\n";
            //of << "double" << "\n";
            //for (int i = 0; i < 8; i++) {
                //ptr = (unsigned long long *) &mr.s_wgt[i];
                //bs = std::bitset<64>(*ptr);
                //of << bs << "\n";
            //}
            //of << "\n";

            //// wgt
            //of << "wgt\n";
            //of << 16 << "\n";
            //of << "double" << "\n";
            //for (int i = 0; i < 16; i++) {
                //ptr = (unsigned long long *) &mr.wgt[i];
                //bs = std::bitset<64>(*ptr);
                //of << bs << "\n";
            //}
            //of << "\n";

            of.close();
            if (err.failed()) std::exit(EXIT_FAILURE);

#ifdef BOOKLEAF_MPI_SUPPORT
        }
        TYPH_Barrier();
    }
#endif
}
#endif // BOOKLEAF_DEBUG

} // namespace setup
} // namespace bookleaf
