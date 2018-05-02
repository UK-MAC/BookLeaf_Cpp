#ifndef BOOKLEAF_DIFF_TRIM_H
#define BOOKLEAF_DIFF_TRIM_H

#include <string>
#include <algorithm>



namespace bookleaf_diff {

inline void ltrim_ip(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}

inline void rtrim_ip(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

inline void trim_ip(std::string &s) {
    ltrim_ip(s);
    rtrim_ip(s);
}

inline std::string ltrim(std::string s) {
    ltrim_ip(s);
    return s;
}

inline std::string rtrim(std::string s) {
    rtrim_ip(s);
    return s;
}

inline std::string trim(std::string s) {
    trim_ip(s);
    return s;
}

} // namespace bookleaf_diff



#endif // BOOKLEAF_DIFF_TRIM_H
