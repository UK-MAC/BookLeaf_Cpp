#include "packages/setup/config.h"

#include "common/error.h"



namespace bookleaf {
namespace setup {

void
rationalise(setup::Config &setup, Error &err)
{
    for (Shape const &shape : setup.shapes)
    {
        shape.rationalise(err);
        if (err.failed()) return;
    }
}

} // namespace setup
} // namespace bookleaf
