#ifndef BOOKLEAF_DIFF_DATA_DUMP_H
#define BOOKLEAF_DIFF_DATA_DUMP_H

#include <string>
#include <fstream>
#include <map>
#include <vector>



namespace bookleaf_diff {

union DiffOptVal {
    bool   flag;
    int    ival;
    double rval;
};

enum class DiffOpt : int {
    IGNORE_OFF_BY_ONE,
    MULTIMATERIAL,
    EPSILON
};

typedef std::map<DiffOpt, DiffOptVal> DiffOpts;

class DataDump
{
public:
    typedef std::size_t size_type;

    /** \brief A data dump consists of several 'array' dumps. */
    struct Array
    {
        static int           constexpr IINIT = -2000000000;
        static double        constexpr RINIT = -2.e12;
        static unsigned char constexpr ZINIT = 0;

        std::string name;   //!< Array name
        size_type   size;   //!< Array size
        std::string type;   //!< Array type
        void       *data;   //!< Array values
    };

public:
    /** \brief Return the number of array entries. */
    int size() const { return arrays.size(); }

    /** \brief Get an array entry by index. */
    Array const &operator[](int i) const { return arrays[i]; }
    Array       &operator[](int i)       { return arrays[i]; }

    /** \brief Get an array entry by name. */
    Array const &get(std::string name) const;
    Array       &get(std::string name);

    /** \brief Read in a data dump file. */
    void
    read(std::string filename);

    /** \brief Compare two data dump instances for equality. */
    bool
    diff(DataDump const &rhs, DiffOpts const &diff_opts = DiffOpts());

private:
    std::vector<Array> arrays;  //!< Store loaded data
};

} // namespace bookleaf_diff



#endif // BOOKLEAF_DIFF_DATA_DUMP_H
