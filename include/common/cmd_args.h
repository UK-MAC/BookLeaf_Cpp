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
#ifndef BOOKLEAF_COMMON_CMD_ARGS_H
#define BOOKLEAF_COMMON_CMD_ARGS_H

#include <string>



namespace bookleaf {

/** \brief Read command-line arguments. */
class CmdArgs
{
public:
    static bool     init(int argc, char **argv);
    static CmdArgs *get();
    static void     kill();

    std::string input_deck_file;    // Input deck file
    bool        overwrite_dumps;    // Overwrite existing dump files?
    bool        print_bindings;     // Print processor bindings

#ifdef BOOKLEAF_DEBUG
    // Debug options
    int         timestep_cap;       // Stop after at most this many timesteps
    bool        dump_mesh;
    bool        dump_init;
    bool        dump_getdt;
    bool        dump_lagstep;
    bool        dump_alestep;
    bool        dump_final;
#endif

private:
    static CmdArgs *singleton;

    CmdArgs();
    ~CmdArgs();

    void setDefaults();
    bool read(int argc, char **argv);

    static void printUsage(std::string invocation);
};



#define INIT_CMD_ARGS(_argc, _argv) CmdArgs::init(_argc, argv)
#define KILL_CMD_ARGS()             CmdArgs::kill()

#define CMD_ARGS (*CmdArgs::get())

} // namespace bookleaf



#endif // BOOKLEAF_COMMON_CMD_ARGS_H
