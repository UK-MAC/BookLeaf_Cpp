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
#ifndef BOOKLEAF_PACKAGES_IO_DRIVER_SILO_IO_DRIVER_H
#define BOOKLEAF_PACKAGES_IO_DRIVER_SILO_IO_DRIVER_H

#include <string>
#include <vector>

#include "common/view.h"
#include "packages/io/driver/io_driver.h"



struct DBfile;

namespace bookleaf {

struct Error;
class TimerControl;
class DataControl;
namespace comms { struct Comm; }
struct Sizes;

namespace io_utils {
struct Labels;
}

namespace io {

class SiloIODriver : public IODriver
{
private:
    static std::string const DATA_NAME;
    static std::string const MESH;
    static std::string const HEADER_FILE;
    static std::string const MATERIAL;
    static std::string const NODE_LIST;

public:
    explicit SiloIODriver();
    ~SiloIODriver();

    void
    dump(
            std::string dir,
            io_utils::Labels const &io,
            Sizes const &sizes,
            DataControl const &data,
            comms::Comm const &comm,
            TimerControl &timers,
            Error &err);

private:
    std::vector<int>    mat_nos;    // Material numbers, used when dumping
    std::vector<char *> mat_names;  // Material names, used when dumping

    // Coordinate writing
    void writeHeader(std::string dir, DataControl const &data,
            comms::Comm const &comm, Error &err);
    void writeData(std::string dir, DataControl const &data, Sizes const &sizes,
            comms::Comm const &comm, Error &err);

    // Write header
    void writeHeaderMesh(DBfile *header,
            std::vector<std::string> const &header_strings,
            comms::Comm const &comm, Error &err);
    void writeHeaderMaterials(DBfile *header,
            std::vector<std::string> const &header_strings,
            comms::Comm const &comm, Error &err);
    void writeHeaderVariable(DBfile *header,
            std::vector<std::string> const &header_strings, std::string var,
            comms::Comm const &comm, Error &err);

    // Write data
    void
    writeDataMesh(
            DBfile *fdata,
            Sizes const &sizes,
            DataControl const &data,
            Error &err);

    void
    writeDataMaterial(
            DBfile *fdata,
            Sizes const &sizes,
            ConstView<int, VarDim> mat_list,
            Error &err);

    void
    writeDataMaterial(
            DBfile *fdata,
            Sizes const &sizes,
            ConstView<int, VarDim>    mat_list,
            ConstView<int, VarDim>    mix_next,
            ConstView<int, VarDim>    mix_mat,
            ConstView<int, VarDim>    mix_zone,
            ConstView<double, VarDim> mix_vf,
            Error &err);

    void
    writeDataVariable(
            DBfile *fdata,
            std::string var_name,
            ConstView<double, VarDim> var,
            int var_len,
            bool zcentre,
            Error &err);

    void
    writeDataVariable(
            DBfile *fdata,
            std::string var_name,
            ConstView<double, VarDim> var,
            ConstView<double, VarDim> mix_var,
            int var_len,
            int mix_var_len,
            bool zcentre, Error &err);

    void
    writeDataVariable(
            DBfile *fdata,
            std::string var_name,
            ConstView<int, VarDim> var,
            int var_len,
            bool zcentre,
            Error &err);

    // Create files
    DBfile *createHeader(std::string dir, std::string filename, Error &err);
    DBfile *createData(std::string dir, std::string filename,
            comms::Comm const &comm, Error &err);

    // Populate mat_{nos,names}
    void getMaterialInfo(io_utils::Labels const &io);
    void freeMaterialInfo();
};

} // namespace io
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_IO_DRIVER_SILO_IO_DRIVER_H
