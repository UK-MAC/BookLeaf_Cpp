#include "common/view.h"

#include "packages/setup/kernel/set_flags.h"



int
main(
        int argc __attribute__((unused)),
        char const *argv[] __attribute__((unused)))
{
    using namespace bookleaf;

    int constexpr NEL = 1000;
    int _flag[NEL];
    View<int, VarDim> flag(_flag, NEL);

    int constexpr FLAG = 1;
    setup::kernel::setFlag(NEL, FLAG, flag);

    // All flags should be set to FLAG
    for (int iel = 0; iel < NEL; iel++) {
        if (flag(iel) != FLAG) return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
