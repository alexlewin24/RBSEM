## require at least autoconf 2.62 ( OMP MACRO )
AC_PREREQ(2.62)

## Process this file with autoconf to produce a configure script.
AC_INIT(rBSEM, m4_esyscmd_s([awk -e '/^Version:/ {print $2}' DESCRIPTION]))

: ${R_HOME=$(R RHOME)}
if test -z "${R_HOME}"; then
    AC_MSG_ERROR([Could not determine R_HOME.])
fi

CXX=$(${R_HOME}/bin/R CMD config CXX)
CXXFLAGS=$("${R_HOME}/bin/R" CMD config CXXFLAGS)

AC_LANG(C++)
AC_PROG_CXX

AC_OPENMP

AC_SUBST(OPENMP_FLAG,"$OPENMP_CXXFLAGS")

AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
