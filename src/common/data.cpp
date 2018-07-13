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
#include "common/data.h"



namespace bookleaf {

bool            Data::partial_sync         = false;
int            *Data::host_sync_send_idx   = nullptr;
int            *Data::device_sync_send_idx = nullptr;
int            *Data::host_sync_recv_idx   = nullptr;
int            *Data::device_sync_recv_idx = nullptr;
Data::size_type Data::sync_send_nidx       = 0;
Data::size_type Data::sync_recv_nidx       = 0;
unsigned char  *Data::host_sync_buf        = nullptr;
unsigned char  *Data::device_sync_buf      = nullptr;
Data::size_type Data::sync_buf_size        = 0;

void
Data::deallocate()
{
    if (isAllocated()) {
        free(host_ptr);
        host_ptr = nullptr;

        dims[0] = 0;
        dims[1] = 0;

        typh_quant_id = -1;
        allocated_T_size = 0;
        len = 0;
        name = "";
        id = DataID::COUNT;
    }
}



void
Data::initPartialSync(
        int *send_indices __attribute__((unused)),
        int *recv_indices __attribute__((unused)),
        int nsend __attribute__((unused)),
        int nrecv __attribute__((unused)))
{
    // No-op in reference version
}



void
Data::killPartialSync()
{
    // No-op in reference version
}



void
Data::syncHost(bool allow_partial __attribute__((unused))) const
{
    // No-op in reference version
}



void
Data::syncDevice(bool allow_partial __attribute__((unused))) const
{
    // No-op in reference version
}

} // namespace bookleaf
