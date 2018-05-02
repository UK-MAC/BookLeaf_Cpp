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
    int _test[NEL];
    View<int, VarDim> flag(_flag, NEL);
    View<int, VarDim> test(_test, NEL);

    // Initialise flag
    int constexpr FLAG_ON  = 1;
    int constexpr FLAG_OFF = 0;
    for (int iel = 0; iel < NEL; iel++) {
        flag(iel) = FLAG_OFF;
    }

    // Set alternating test values.
    int constexpr TEST_ON  = 1;
    int constexpr TEST_OFF = 0;
    for (int iel = 0; iel < NEL; iel++) {
        test(iel) =  (iel % 2 == 0) ? TEST_ON : TEST_OFF;
    }

    setup::kernel::setFlagIf(NEL, FLAG_ON, TEST_ON, test, flag);

    // Even flags should be set to FLAG_ON, odd to FLAG_OFF
    for (int iel = 0; iel < NEL; iel++) {
        bool const correct =
            (iel % 2 == 0) ? (flag(iel) == FLAG_ON) : (flag(iel) == FLAG_OFF);

        if (!correct) return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
