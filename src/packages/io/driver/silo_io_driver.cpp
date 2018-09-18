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
#include "packages/io/driver/silo_io_driver.h"

#include <cstring>
#include <numeric>
#include <cassert>

#include <silo.h>

#ifdef BOOKLEAF_MPI_SUPPORT
#include <typhon.h>
#endif

#include "common/error.h"
#include "common/sizes.h"
#include "common/data_control.h"
#include "common/constants.h"
#include "common/cmd_args.h"
#include "utilities/misc/string_utils.h"
#include "common/timer_control.h"
#include "packages/setup/indicators.h"
#include "utilities/io/config.h"

#include "utilities/comms/config.h"



namespace bookleaf {
namespace io {

std::string const SiloIODriver::DATA_NAME   = "domain";
std::string const SiloIODriver::MESH        = "Mesh";
std::string const SiloIODriver::HEADER_FILE = "Top.silo";
std::string const SiloIODriver::MATERIAL    = "Material";
std::string const SiloIODriver::NODE_LIST   = "NodeList";



SiloIODriver::SiloIODriver()
{
}



SiloIODriver::~SiloIODriver()
{
    freeMaterialInfo();
}



void
SiloIODriver::dump(
        std::string dir,
        io_utils::Labels const &io,
        Sizes const &sizes,
        DataControl const &data,
        comms::Comm const &comm,
        TimerControl &timers,
        Error &err)
{
    ScopedTimer st(timers, TimerID::IO);

    // Make directory
    makeDirectory(dir, err);
    if (err.failed()) return;

    // Populate mat_nos and mat_names for use during dump
    getMaterialInfo(io);

    // Write header file on master
    if (comm.zmproc) {
      writeHeader(dir, data, comm, err);
      if (err.failed()) return;
    }

#ifdef BOOKLEAF_MPI_SUPPORT
    TYPH_Barrier();
#endif

    // Write data file on each domain
    writeData(dir, data, sizes, comm, err);
    if (err.failed()) return;

    // Create symbolic link on master
    if (comm.zmproc) {
        makeSymbolicLink(dir + "/" + HEADER_FILE, dir + ".silo", err);
        if (err.failed()) return;
    }
}



void
SiloIODriver::writeHeader(
        std::string dir,
        DataControl const &data,
        comms::Comm const &comm,
        Error &err)
{
    // Create header file
    DBfile *header = createHeader(dir, HEADER_FILE, err);
    if (err.failed()) return;

    auto close = [header, &err]() {
        if (DBClose(header) != 0) {
            err.fail("ERROR: failed to close header file");
        }
    };

    // Create header strings
    std::vector<std::string> header_strings;
    for (int i = 0; i < comm.nproc; i++) {
        std::string num = std::to_string(i+1000);
        header_strings.push_back(dir + "/" + DATA_NAME + num.substr(1) + ".silo");
    }

    // Write mesh header
    writeHeaderMesh(header, header_strings, comm, err);
    if (err.failed()) {
        close();
        return;
    }

    // Write material header
    writeHeaderMaterials(header, header_strings, comm, err);
    if (err.failed()) {
        close();
        return;
    }

    // Write variables header
    std::vector<DataID> vars {
        DataID::ELDENSITY,
        DataID::ELENERGY,
        DataID::ELPRESSURE,
        DataID::NDU,
        DataID::NDV,
        DataID::IELREG
    };

    for (auto var : vars) {
        writeHeaderVariable(header, header_strings, data[var].getName(), comm, err);
        if (err.failed()) {
            close();
            return;
        }
    }

    close();
}



void
SiloIODriver::writeData(
        std::string dir,
        DataControl const &data,
        Sizes const &sizes,
        comms::Comm const &comm,
        Error &err)
{
    // Create data file on each PE
    DBfile *fdata = createData(dir, DATA_NAME, comm, err);
    if (err.failed()) return;

    auto close = [fdata, &err]() {
        if (DBClose(fdata) != 0) {
            err.fail("ERROR: failed to close data file");
        }
    };

    // Write mesh
    writeDataMesh(fdata, sizes, data, comm, err);
    if (err.failed()) {
        close();
        return;
    }

    // Write material and thermodynamic data, handling mixed cells
    if (sizes.ncp > 0) {
        auto elmat      = data[DataID::IELMAT].chost<int, VarDim>();
        auto eldensity  = data[DataID::ELDENSITY].chost<double, VarDim>();
        auto elenergy   = data[DataID::ELENERGY].chost<double, VarDim>();
        auto elpressure = data[DataID::ELPRESSURE].chost<double, VarDim>();
        auto cpdensity  = data[DataID::CPDENSITY].chost<double, VarDim>();
        auto cpenergy   = data[DataID::CPENERGY].chost<double, VarDim>();
        auto cppressure = data[DataID::CPPRESSURE].chost<double, VarDim>();
        auto mxel     = data[DataID::IMXEL].chost<int, VarDim>();
        auto mxfcp    = data[DataID::IMXFCP].chost<int, VarDim>();
        auto mxncp    = data[DataID::IMXNCP].chost<int, VarDim>();
        auto cpmat    = data[DataID::ICPMAT].chost<int, VarDim>();
        auto cpnext   = data[DataID::ICPNEXT].chost<int, VarDim>();
        auto frvolume = data[DataID::FRVOLUME].chost<double, VarDim>();

        // Convert multi-material data to SILO format
        std::vector<int>   _scratch(data[DataID::ISCRATCH11].size());
        std::vector<int> _cpscratch(data[DataID::ICPSCRATCH11].size());
        std::vector<int> _cpnext1off(sizes.ncp);

        View<int, VarDim>   scratch(  _scratch.data(),   _scratch.size());
        View<int, VarDim> cpscratch(_cpscratch.data(), _cpscratch.size());
        View<int, VarDim> cpnext1off(_cpnext1off.data(), sizes.ncp);

        // Copy elmat to scratch
        for (int i = 0; i < sizes.nel; i++) {
            scratch[i] = elmat[i];
        }

        {
            int const nmx = sizes.nmx;

            auto &cpel = cpscratch;
            auto &elmat = scratch;

            assert(mxel.size() == mxfcp.size());
            assert(mxel.size() == mxncp.size());
            assert(cpnext.size() == cpel.size());

            // set mixed element elmat to negative component indices
            // set back-indices for each component to parent element
            for (int imix = 0; imix < nmx; imix++) {
                int iel = mxel(imix);
                int icp = mxfcp(imix);
                elmat(iel) = -(icp+1);
                for (int ii = 0; ii < mxncp(imix); ii++) {
                    cpel(icp) = iel;
                    icp = cpnext(icp);
                }
            }

            // cpnext indices need to be 1-origin
            for (int icp = 0; icp < sizes.ncp; icp++) {
                cpnext1off(icp) = cpnext(icp) + 1;
            }
        }

        // Write material
        writeDataMaterial(fdata, sizes, scratch, cpnext1off, cpmat, cpscratch,
                frvolume, err);
        if (err.failed()) {
            close();
            return;
        }

        // Write thermodynamic variables
        writeDataVariable(fdata, data[DataID::ELDENSITY].getName(), eldensity,
                cpdensity, sizes.nel, sizes.ncp, true, err);
        writeDataVariable(fdata, data[DataID::ELENERGY].getName(), elenergy,
                cpenergy, sizes.nel, sizes.ncp, true, err);
        writeDataVariable(fdata, data[DataID::ELPRESSURE].getName(), elpressure,
                cppressure, sizes.nel, sizes.ncp, true, err);
        if (err.failed()) {
            close();
            return;
        }

    // Write pure material and thermodynamic data
    } else {
        auto elmat      = data[DataID::IELMAT].chost<int, VarDim>();
        auto eldensity  = data[DataID::ELDENSITY].chost<double, VarDim>();
        auto elenergy   = data[DataID::ELENERGY].chost<double, VarDim>();
        auto elpressure = data[DataID::ELPRESSURE].chost<double, VarDim>();

        // Write material
        writeDataMaterial(fdata, sizes, elmat, err);
        if (err.failed()) {
            close();
            return;
        }

        // Write thermodynamic variables
        writeDataVariable(fdata, data[DataID::ELDENSITY].getName(), eldensity,
                sizes.nel, true, err);
        writeDataVariable(fdata, data[DataID::ELENERGY].getName(), elenergy,
                sizes.nel, true, err);
        writeDataVariable(fdata, data[DataID::ELPRESSURE].getName(), elpressure,
                sizes.nel, true, err);
        if (err.failed()) {
            close();
            return;
        }
    }

    // Write pure variables
    auto elreg = data[DataID::IELREG].chost<int, VarDim>();
    auto ndu   = data[DataID::NDU].chost<double, VarDim>();
    auto ndv   = data[DataID::NDV].chost<double, VarDim>();

    writeDataVariable(fdata, data[DataID::NDU].getName(), ndu, sizes.nnd,
            false, err);
    writeDataVariable(fdata, data[DataID::NDV].getName(), ndv, sizes.nnd,
            false, err);
    writeDataVariable(fdata, data[DataID::IELREG].getName(), elreg, sizes.nel,
            true, err);

    close();
}



void
SiloIODriver::writeHeaderMesh(
        DBfile *header,
        std::vector<std::string> const &header_strings,
        comms::Comm const &comm,
        Error &err)
{
    // Setup mesh info
    std::vector<char *> mesh_names(comm.nproc);
    for (int i = 0; i < comm.nproc; i++) {
        std::string str(header_strings[i] + ":" + MESH);
        mesh_names[i] = (char *) calloc(str.size() + 1, sizeof(char));
        strncpy(mesh_names[i], str.c_str(), str.size());
    }

    std::vector<int> mesh_types(comm.nproc);
    std::fill(mesh_types.begin(), mesh_types.end(), DB_UCDMESH);

    // Write to header file
    if (DBPutMultimesh(
                header,             // File handle
                MESH.c_str(),       // Name of the multi-block mesh object
                comm.nproc,         // Number of mesh pieces (blocks) in object
                mesh_names.data(),  // Name of each mesh block
                mesh_types.data(),  // Type of each mesh block
                nullptr             // Option list (not applicable)
            ) != 0) {

        err.fail("ERROR: failed to put multimesh");
    }

    // Clean up
    for (char *mesh_name : mesh_names) { free(mesh_name); }
}



void
SiloIODriver::writeHeaderMaterials(
        DBfile *header,
        std::vector<std::string> const &header_strings,
        comms::Comm const &comm,
        Error &err)
{
    // Create option list
    DBoptlist *opt_list = nullptr;
    if ((opt_list = DBMakeOptlist(4)) == nullptr) {
        err.fail("ERROR: failed to create option list");
        return;
    }

    // Setup data to store as options
    int nmat = mat_nos.size();

    // Add the options
    int ierr = 0;
    ierr |= DBAddOption(opt_list, DBOPT_NMATNOS, &nmat);
    ierr |= DBAddOption(opt_list, DBOPT_MATNOS, mat_nos.data());
    ierr |= DBAddOption(opt_list, DBOPT_MATNAMES, mat_names.data());
    ierr |= DBAddOption(opt_list, DBOPT_MMESH_NAME,
            const_cast<char *>(MESH.c_str()));
    if (ierr != 0) {
        err.fail("ERROR: failed to populate option list");
        DBFreeOptlist(opt_list);
        return;
    }

    // Write the header entry
    std::vector<char *> db_mat_names(comm.nproc);
    for (int i = 0; i < comm.nproc; i++) {
        std::string str(header_strings[i] + ":" + MATERIAL);
        db_mat_names[i] = (char *) calloc(str.size() + 1, sizeof(char));
        strncpy(db_mat_names[i], str.c_str(), str.size());
    }

    if (DBPutMultimat(
                header,                 // File handle
                MATERIAL.c_str(),       // Name of the material data object
                comm.nproc,             //
                db_mat_names.data(),    //
                opt_list                // Option list
            ) != 0) {

        err.fail("ERROR: failed to put multimaterial");
        DBFreeOptlist(opt_list);
        for (char *mat_name : db_mat_names) { free(mat_name); }
        return;
    }

    // Clean up
    if (DBFreeOptlist(opt_list) != 0) {
        err.fail("ERROR: failed to free option list");
    }

    for (char *mat_name : db_mat_names) { free(mat_name); }
}



void
SiloIODriver::writeHeaderVariable(
        DBfile *header,
        std::vector<std::string> const &header_strings,
        std::string var,
        comms::Comm const &comm,
        Error &err)
{
    std::string svar;
    if (var.size() >= 3) svar = var.substr(2);
    else {
        err.fail("ERROR: invalid variable name " + var);
        return;
    }

    // Setup variable info
    std::vector<char *> var_names(comm.nproc);
    for (int i = 0; i < comm.nproc; i++) {
        std::string str(header_strings[i] + ":" + svar);
        var_names[i] = (char *) calloc(str.size() + 1, sizeof(char));
        strncpy(var_names[i], str.c_str(), str.size());
    }

    std::vector<int> var_types(comm.nproc);
    std::fill(var_types.begin(), var_types.end(), DB_UCDVAR);

    // Write to header file
    if (DBPutMultivar(
                header,             // File handle
                svar.c_str(),       // Name of the multi-block variable
                comm.nproc,         //
                var_names.data(),   //
                var_types.data(),   //
                nullptr             // Option list (not applicable)
            ) != 0) {

        err.fail("ERROR: failed to put multivariable");
    }

    // Clean up
    for (char *var_name : var_names) { free(var_name); }
}



void
SiloIODriver::writeDataMesh(
        DBfile *fdata,
        Sizes const &sizes,
        DataControl const &data,
        comms::Comm const &comm,
        Error &err)
{
    using constants::NCORN;

    int constexpr NSHAPE = 1;
    int constexpr NSIZE  = 4;

    // Create option list
    DBoptlist *opt_list = nullptr;
    if ((opt_list = DBMakeOptlist(1)) == nullptr) {
        err.fail("ERROR: failed to create option list");
        return;
    }

    // Write zone list
    auto node_list = data[DataID::IELND].chost<int, VarDim, NCORN>();
    int *node_list_ptr = const_cast<int *>(node_list.data());
    int len_node_list = node_list.size();

    int *zone_glob_ptr = nullptr;
#ifdef BOOKLEAF_MPI_SUPPORT
    if (comm.nproc > 1) {
        auto zone_glob = data[DataID::IELLOCGLOB].chost<int, VarDim>();
        zone_glob_ptr = const_cast<int *>(zone_glob.data());
    } else
#endif
    {
        zone_glob_ptr = new int[sizes.nel];
        std::iota(zone_glob_ptr, zone_glob_ptr+sizes.nel, 0);
    }

    int shape_type  = DB_ZONETYPE_QUAD;
    int shape_size  = NSIZE;
    int shape_count = sizes.nel;

    int ierr = 0;
    ierr |= DBAddOption(opt_list, DBOPT_ZONENUM, zone_glob_ptr);
    if (ierr != 0) {
        err.fail("ERROR: failed to populate option list");
        DBFreeOptlist(opt_list);
        return;
    }

    if (DBPutZonelist2(
                fdata,              // File handle
                NODE_LIST.c_str(),  // Name of the zonelist structure
                sizes.nel,          // Number of zones in associated mesh
                2,                  // Number of spatial dims in ass. mesh
                node_list_ptr,      // Node indices describing mesh zones
                len_node_list,      // Node list length
                0,                  // Origin for indices in node list (0/1)
                0,                  // # ghost zones at start of node list
                0,                  // # ghost zones at end of node list
                &shape_type,        // Type of each zone shape
                &shape_size,        // # nodes per zone shape
                &shape_count,       // # zones having each shape
                NSHAPE,             // # different zone shapes
                opt_list            // Option list
            ) != 0) {

        err.fail("ERROR: failed to put zonelist");
        return;
    }

    // Clean up
    if (DBFreeOptlist(opt_list) != 0) {
        err.fail("ERROR: failed to free option list");
    }
    opt_list = nullptr;

    if (comm.nproc == 1) {
        delete[] zone_glob_ptr;
    }

    // Prepare coordinate pointers
    auto ndx = data[DataID::NDX].chost<double, VarDim>();
    auto ndy = data[DataID::NDY].chost<double, VarDim>();
    double const *coords[2] = { ndx.data(), ndy.data() };

    // Create option list
    if ((opt_list = DBMakeOptlist(4)) == nullptr) {
        err.fail("ERROR: failed to create option list");
        return;
    }

    std::string xlabel = "X";
    char *pxlabel = const_cast<char *>(xlabel.c_str());
    std::string ylabel = "Y";
    char *pylabel = const_cast<char *>(ylabel.c_str());

    int coordsys = DB_CARTESIAN;

    int *node_glob_ptr = nullptr;
#ifdef BOOKLEAF_MPI_SUPPORT
    if (comm.nproc > 1) {
        auto node_glob = data[DataID::INDLOCGLOB].chost<int, VarDim>();
        node_glob_ptr = const_cast<int *>(node_glob.data());
    } else
#endif
    {
        node_glob_ptr = new int[sizes.nnd];
        std::iota(node_glob_ptr, node_glob_ptr+sizes.nnd, 0);
    }

    ierr = 0;
    ierr |= DBAddOption(opt_list, DBOPT_XLABEL, pxlabel);
    ierr |= DBAddOption(opt_list, DBOPT_YLABEL, pylabel);
    ierr |= DBAddOption(opt_list, DBOPT_COORDSYS, &coordsys);
    ierr |= DBAddOption(opt_list, DBOPT_NODENUM, node_glob_ptr);
    if (ierr != 0) {
        err.fail("ERROR: failed to populate option list");
        DBFreeOptlist(opt_list);
        return;
    }

    // Write mesh
    if (DBPutUcdmesh(
                fdata,              // File handle
                MESH.c_str(),       // Mesh name
                2,                  // Number of spatial dimensions
                nullptr,            // Coordinate dimension names (ignored)
                coords,             // Coordinates
                sizes.nnd,          // Number of nodes in the mesh
                sizes.nel,          // Number of zones in the mesh
                NODE_LIST.c_str(),  // Name of associated zonelist
                nullptr,            // Name of associated facelist
                DB_DOUBLE,          // Coords datatype
                opt_list            // Option list
            ) != 0) {

        err.fail("ERROR: failed to put ucd mesh");
    }

    // Clean up
    if (DBFreeOptlist(opt_list) != 0) {
        err.fail("ERROR: failed to free option list");
    }

    if (comm.nproc == 1) {
        delete[] node_glob_ptr;
    }
}



void
SiloIODriver::writeDataMaterial(
        DBfile *fdata,
        Sizes const &sizes,
        ConstView<int, VarDim> mat_list,
        Error &err)
{
    // Create option list
    DBoptlist *opt_list = nullptr;
    if ((opt_list = DBMakeOptlist(2)) == nullptr) {
        err.fail("ERROR: failed to create option list");
        return;
    }

    // Fill option list
    int origin = 0;

    int ierr = 0;
    ierr |= DBAddOption(opt_list, DBOPT_ORIGIN, &origin);
    ierr |= DBAddOption(opt_list, DBOPT_MATNAMES, mat_names.data());
    if (ierr != 0) {
        err.fail("ERROR: failed to populate option list");
        DBFreeOptlist(opt_list);
        return;
    }

    // Prepare other material data
    int nel = sizes.nel;
    int *_mat_list = const_cast<int *>(mat_list.data());

    // Write material data object
    if (DBPutMaterial(
                fdata,              // File handle
                MATERIAL.c_str(),   // Name of the material data object
                MESH.c_str(),       // Name of associated mesh
                mat_nos.size(),     // Number of materials
                mat_nos.data(),     // Material numbers
                _mat_list,          // Specify material(s) per zone
                &nel,               // Dimensions of mat_list
                1,                  // Number of dimensions in mat_list
                nullptr,            // Mixed material arrays
                nullptr,            //          "
                nullptr,            //          "
                nullptr,            //          "
                0,                  // Length of mixed material arrays
                DB_DOUBLE,          // Volume fraction data type
                opt_list            // Option list
            ) != 0) {

        err.fail("ERROR: failed to put material");
    }

    // Clean up
    if (DBFreeOptlist(opt_list) != 0) {
        err.fail("ERROR: failed to free option list");
    }
}



void
SiloIODriver::writeDataMaterial(
        DBfile *fdata,
        Sizes const &sizes,
        ConstView<int, VarDim>    mat_list,
        ConstView<int, VarDim>    mix_next,
        ConstView<int, VarDim>    mix_mat,
        ConstView<int, VarDim>    mix_zone,
        ConstView<double, VarDim> mix_vf,
        Error &err)
{
    // Create option list
    DBoptlist *opt_list = nullptr;
    if ((opt_list = DBMakeOptlist(2)) == nullptr) {
        err.fail("ERROR: failed to create option list");
        return;
    }

    // Fill option list
    int origin = 0;

    int ierr = 0;
    ierr |= DBAddOption(opt_list, DBOPT_ORIGIN, &origin);
    ierr |= DBAddOption(opt_list, DBOPT_MATNAMES, mat_names.data());
    if (ierr != 0) {
        err.fail("ERROR: failed to populate option list");
        DBFreeOptlist(opt_list);
        return;
    }

    // Prepare other material data
    int nel = sizes.nel;

    int *_mat_list = const_cast<int *>(mat_list.data());

    int  *_mix_next = const_cast<int *>(mix_next.data());
    int   *_mix_mat = const_cast<int *>(mix_mat.data());
    int  *_mix_zone = const_cast<int *>(mix_zone.data());
    double *_mix_vf = const_cast<double *>(mix_vf.data());

    // Write material data object
    if (DBPutMaterial(
                fdata,              // File handle
                MATERIAL.c_str(),   // Name of the material data object
                MESH.c_str(),       // Name of associated mesh
                mat_nos.size(),     // Number of materials
                mat_nos.data(),     // Material numbers
                _mat_list,          // Specify material(s) per zone
                &nel,               // Dimensions of mat_list
                1,                  // Number of dimensions in mat_list
                _mix_next,          // Component next indices
                _mix_mat,           // Component material indices
                _mix_zone,          // Back-indices for component elements
                _mix_vf,            // Component volume fractions
                sizes.ncp,          // Length of mixed material arrays
                DB_DOUBLE,          // Volume fraction data type
                opt_list            // Option list
            ) != 0) {

        err.fail("ERROR: failed to put material");
    }

    // Clean up
    if (DBFreeOptlist(opt_list) != 0) {
        err.fail("ERROR: failed to free option list");
    }
}



void
SiloIODriver::writeDataVariable(
        DBfile *fdata,
        std::string var_name,
        ConstView<double, VarDim> var,
        int var_len,
        bool zcentre,
        Error &err)
{
    int centre = zcentre ? DB_ZONECENT : DB_NODECENT;
    var_name = var_name.substr(2);

    double *_var = const_cast<double *>(var.data());

    if (DBPutUcdvar1(
                fdata,              // File handle
                var_name.c_str(),   // Name of the variable
                MESH.c_str(),       // Name of the associated mesh
                _var,               // Variable values
                var_len,            // Variable values count
                nullptr,            // Mixed data values
                0,                  // Mixed data values count
                DB_DOUBLE,          // Data type
                centre,             // Centreing of sub-variables on ass. mesh
                nullptr             // Option list (not applicable)
            ) != 0) {

        err.fail("ERROR: failed to put ucd mesh variable");
    }
}



void
SiloIODriver::writeDataVariable(
        DBfile *fdata,
        std::string var_name,
        ConstView<double, VarDim> var,
        ConstView<double, VarDim> mix_var,
        int var_len,
        int mix_var_len,
        bool zcentre,
        Error &err)
{
    int centre = zcentre ? DB_ZONECENT : DB_NODECENT;
    var_name = var_name.substr(2);

    double *_var     = const_cast<double *>(var.data());
    double *_mix_var = const_cast<double *>(mix_var.data());

    if (DBPutUcdvar1(
                fdata,              // File handle
                var_name.c_str(),   // Name of the variable
                MESH.c_str(),       // Name of the associated mesh
                _var,               // Variable values
                var_len,            // Variable values count
                _mix_var,           // Mixed data values
                mix_var_len,        // Mixed data values count
                DB_DOUBLE,          // Data type
                centre,             // Centreing of sub-variables on ass. mesh
                nullptr             // Option list (not applicable)
            ) != 0) {

        err.fail("ERROR: failed to put ucd mesh variable");
    }
}



void
SiloIODriver::writeDataVariable(
        DBfile *fdata,
        std::string var_name,
        ConstView<int, VarDim> var,
        int var_len,
        bool zcentre,
        Error &err)
{
    int centre = zcentre ? DB_ZONECENT : DB_NODECENT;
    var_name = var_name.substr(2);

    int *_var = const_cast<int *>(var.data());

    if (DBPutUcdvar1(
                fdata,              // File handle
                var_name.c_str(),   // Name of the variable
                MESH.c_str(),       // Name of the associated mesh
                _var,               // Variable values
                var_len,            // Variable values count
                nullptr,            // Mixed data values
                0,                  // Mixed data values count
                DB_INT,             // Data type
                centre,             // Centreing of sub-variables on ass. mesh
                nullptr             // Option list (not applicable)
            ) != 0) {

        err.fail("ERROR: failed to put ucd mesh variable");
    }
}



DBfile *
SiloIODriver::createHeader(std::string dir, std::string filename, Error &err)
{
    std::string path = trim(dir) + "/" + trim(filename);

    // FIXME(timrlaw): Silo doesn't seem to handle NOCLOBBER correctly.
    if (exists(path)) {
        if (CMD_ARGS.overwrite_dumps) {
            removeFile(path,err);
        } else {
            err.fail("ERROR: silo header file already exists");
            return nullptr;
        }
    }

    int const mode = DB_CLOBBER;//CMD_ARGS.overwrite_dumps ? DB_CLOBBER : DB_NOCLOBBER;

    DBfile *file = nullptr;
    if ((file = DBCreate(
                    path.c_str(),
                    mode,
                    DB_LOCAL,
                    "SILO file from Bookleaf",
                    DB_HDF5)) == nullptr) {

        err.fail("ERROR: failed to create silo header file");
        return nullptr;
    }

    return file;
}



DBfile *
SiloIODriver::createData(
        std::string dir,
        std::string data_name,
        comms::Comm const &comm,
        Error &err)
{
    std::string srank = std::to_string(comm.rank + 1000);
    std::string path = trim(dir) + "/" + trim(data_name) + srank.substr(1) +
        ".silo";

    // FIXME(timrlaw): Silo doesn't seem to handle NOCLOBBER correctly.
    if (exists(path)) {
        if (CMD_ARGS.overwrite_dumps) {
            removeFile(path,err);
        } else {
            err.fail("ERROR: silo header file already exists");
            return nullptr;
        }
    }

    int const mode = DB_CLOBBER;//CMD_ARGS.overwrite_dumps ? DB_CLOBBER : DB_NOCLOBBER;

    DBfile *file = nullptr;
    if ((file = DBCreate(
                    path.c_str(),
                    mode,
                    DB_LOCAL,
                    "SILO file from bookleaf",
                    DB_HDF5)) == nullptr) {

        err.fail("ERROR: failed to create silo data file");
        return nullptr;
    }

    return file;
}



void
SiloIODriver::getMaterialInfo(io_utils::Labels const &io)
{
    freeMaterialInfo();

    mat_nos.resize(io.smaterials.size());
    std::iota(mat_nos.begin(), mat_nos.end(), 0);

    std::vector<char *> mat_names(io.smaterials.size());
    for (int i = 0; i < (int) mat_names.size(); i++) {
        mat_names[i] = (char *) calloc(
                io.smaterials[i].size() + 1, sizeof(char));
        strncpy(mat_names[i], io.smaterials[i].c_str(),
                io.smaterials[i].size());
    }
}



void
SiloIODriver::freeMaterialInfo()
{
    for (char *mat_name : mat_names) {
        if (mat_name != nullptr) {
            free(mat_name);
            mat_name = nullptr;
        }
    }
}

} // namespace io
} // namespace bookleaf
