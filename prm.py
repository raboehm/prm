#! /usr/bin/env python3
"""Find polynomial roots with multiplicities using mpmath.

Running prm.py will solve about 70 test polynomials and roots, multiplicities,
iterations, execution times, figures of merit and method used.

Bob Boehm 2017
raboehm verizon net
"""

from multiprocessing import Process, cpu_count, Manager
from socket import gethostname
from functools import reduce
from operator import mul
from traceback import extract_stack
import time
import sys
import math as ma
import numpy as np
import mpmath as mp

ZERO = mp.mpf(0)
ONE = mp.mpf(1)
TWO = mp.mpf(2)
THREE = mp.mpf(3)
TEN = mp.mpf(10)
FM01 = "AA2:  n=%d"
FM11 = "BB6:  n=%d;"
FM18 = "EE:  n=%d; itr=%d; jji=%d; kki=%d;"
FM28 = "FF8:  val[%d]=%+20.12e %+20.12e j; acc=%+20.12e %+20.12e j;"
FM39 = "LL:  n=%4i; delta_method=%2i; itr=%4i; dig=%5i; msft=%9.2f; cst=%8.2f;"


def prm(poly, delta_method=None, full=False):
    """Find roots of a polynomial using two multiple precision methods.

    Parameters
    ----------
    poly : list or array like
        coefficients of the polynomial -- size n+1
        integer, float, or multiple precision, real or complex
    delta_method : list
        delta methods to be used in first and second algorithms -- see Notes
        if None, defaults to [1, 3]
    full : boolean
        additional output if true

    y(x) = poly[0]*x^n + poly[1]*x^(n-1) + ... + poly[n-1]*x + poly[n]

    Returns
    -------
    zout : ndarray
        multiple precision ndarray containing the roots of the polynomial
    mlt : list
        multiplicity of zout
    and if full:
    msf : float
        figure of merit of root estimates -- larger negative numbers are better
        log10(sum(abs(polyval(zout))) / n)) / 2
    num_iter : int
        total number of iterations
    tstr : string
        method that provided root estimate
        "sing" = single root algorithm found roots within tolerance first
        "(sing)" = both methods finished, single root algorithm was better
        "mult" = multiple root algorithm found roots within tolerance first
        "(mult)" = both methods finished, multiple root algorithm was better

    Notes
    -----
    The goal is to find the roots of a polynomial with arbitrary complex
    coefficients whose roots may be singular or multiple.  This method (prm)
    starts two separate methods in parallel -- one method converges well when
    all roots are singular, and the second converges better than the first when
    there are multiple roots.

    For polynomials with only singular roots, there are many algorithms that
    will find roots simultaneously with varying degrees of rate of convergence.
    This software (prm_sing) has options for WeierStrass-Durand-Kerner,
    Aberth-Ehrlich, Sakurai-Torii-Sugiura, and Sakurai-Petkovic, but using the
    test polynomials included (prm_test and polygen), seems to show that
    Sakurai-Torii-Sugiura (delta_sakurai) performed generally the best.
    This is the default for this method.  It uses an algorithm found in
    Krishnan-Foskey-Culver-Keyser-Manocha (rootmax11) to generate initial root
    estimates.

    For polynomials with multiple roots, the methods for singular roots
    converge much more slowly.  The method here (prm_mult) uses the fact that,
    for multiplicities greater than one, the polynomial's derivative also has a
    root there.  This second algorithm finds the roots of the derivative and
    checks them in the original polynomial.  It is recursive, so all
    derivatives are checked.  The roots of the derivative are then used as
    initial root estimates for the polynomial (as this seemed better than using
    rootmax11).  The Aberth-Ehrlich (delta_aberth) method seems generally the
    best here, so it is the default.

    If the first algorithm to finish is close enough, the second is stopped.
    Otherwise, the second is allowed to finish and the better solution is
    chosen.

    Running prm.py will solve about 70 test polynomials and roots,
    multiplicities, iterations, execution times, figures of merit and method
    used.  The software still has issues on some of the tests, but it generally
    seems to work OK.

    delta_method
        0    delta_weierstrass  WeierStrass-Durand-Kerner
        1    delta_aberth       Aberth-Ehrlich
        3    delta_sakurai      Sakurai-Torii-Sugiura
        7    delta_petkovic     Sakurai-Petkovic

    empirical results from prm_test and polygen
        prm_mult:  1 is better than 3 & 7, 3 & 7 are equally bad on convergence
                   1 is better than 0 (iterations & time)
        prm_sing:  3 is better than 0 and 1 (iterations & time)
                   3 is better than 7, mostly on time

    References
    ----------
        https://en.wikipedia.org/wiki/Durand%E2%80%93Kerner_method

        https://en.wikipedia.org/wiki/Aberth_method

        T. Sakurai, M.S. Petkovic  1996-Aug-13
        On some simultaneous methods based on Weierstrass correction
        Section 2, Sakurai-Torii-Sugiura, Sakurai-Petkovic

        Krishnan, Foskey, Culver, Keyser, Manocha  2001-Jun-03
        PRECISE:  Efficent Multiprecision Evaluation of Algebraic Roots and
        Predicates for Reliable Geometric Computation
        Section 4.2 Choice of Initial Approximations

    Example
    -------
    >>> mp.mp.dps = 2000
    >>> prm([1, 4, 6, 4, 1], full=True)
    (array([mpc(real='-1.0', imag='0.0'), mpc(real='0.0', imag='0.0'),
           mpc(real='0.0', imag='0.0'), mpc(real='0.0', imag='0.0')],
           dtype=object), [4, 0, 0, 0], -4000, 4, 'mult')

    To Do
    -----
        error handling
        prm has a problem in cygwin's python command line (scripts ok, though):
            python child info fork abort address space needed by properties
            cpython x86 64-cygwin.dll already occupied  (cygwin fork)
    """

    if delta_method is None:
        delta_method = [1, 3]
    dbf = 0
    thrsh = -mp.mp.dps * 0.80   # originally 0.95
    retval = 0
    sleeptime = 0.1
    maxloop = 1000000       # 100000

    poly, rtsat0 = polytrim(poly)
    npoly = len(poly) - 1
    procs = 2

    manager = Manager()
    namspc0 = manager.Namespace()
    namspc1 = manager.Namespace()
    namspcs = [namspc0, namspc1]

    ec_("Z1:  before queue", dbf > 0)
    flgs = [0] * procs  # 0=alive; 1=done, > threshold; 2=done, < threshold

    ec_("Z2:  before process", dbf > 0)
    prms = [prm_mult, prm_sing]
    proc = [None] * procs
    for idx in range(procs):
        namspcs[idx].zout = mpmpc([0] * npoly)
        namspcs[idx].mlt = [0] * npoly
        namspcs[idx].msft = 0
        namspcs[idx].itr = 0
        proc[idx] = Process(target=prm_worker, args=(poly, delta_method[idx],
                                                     prms[idx], namspcs[idx]))
    typ = ["mult", "sing"]
    mlts = [None] * procs
    msfts = [1000000] * procs
    itrs = [None] * procs

    ec_("Z3:  before start", dbf > 0)
    for idx in range(procs):
        proc[idx].start()
    cnt = 0
    # if the first to return meets the solution criterion, kill the other one
    ec_("Z4:  before while", dbf > 0)
    while prod(flgs) == 0 and cnt < maxloop:
        cnt += 1
        for idx in range(procs):
            if max(flgs) == 2:
                break
            ec_("Z5:  while; cnt=%6d; proc[%d].is_alive()=%d;" %
                (cnt, idx, proc[idx].is_alive()), dbf > 1)
            if flgs[idx] == 0 and not proc[idx].is_alive():
                ec_("A1", dbf > 0)
                flgs[idx] = 1
                mlts[idx] = namspcs[idx].mlt
                msfts[idx] = namspcs[idx].msft
                itrs[idx] = namspcs[idx].itr
                ec_("A2 msfts[%d]=%9.2f;" % (idx, msfts[idx]), dbf > 0)
                if msfts[idx] < thrsh:
                    flgs[idx] = 2
                    ec_("A3 flgs[%d]=%d;" % (idx, flgs[idx]), dbf > 0)
                    time.sleep(0.1)
                    zout = namspcs[idx].zout
                    if rtsat0 > 0:
                        zout = np.append(zout, ZERO)
                        mlts[idx] = np.append(mlts[idx], 0)
                        zout, mlts[idx] = apprtsat0(zout, mlts[idx], rtsat0)
                    killprocs(proc)
                    retval = zout, mlts[idx], msfts[idx], itrs[idx], typ[idx]
        if max(flgs) == 2 or prod(flgs) == 1:
            break
        time.sleep(sleeptime)

    ec_("Z6:  after while; cnt=%6d;" % (cnt), dbf > 0)
    if sum(flgs) == 0:      # all processes still running
        ec_("Processes still running:  cnt=%6d/%10d;" % (cnt, maxloop))
        killprocs(proc)
    elif max(flgs) == 1:    # no msfts[idx] < thrsh, return better one
        idx = msfts.index(min(msfts))
        ec_("Cx idx=%d;" % (idx), dbf > 0)
        zout = namspcs[idx].zout
        if rtsat0 > 0:
            zout = np.append(zout, ZERO)
            mlts[idx] = np.append(mlts[idx], 0)
            zout, mlts[idx] = apprtsat0(zout, mlts[idx], rtsat0)
        killprocs(proc)
        retval = zout, mlts[idx], msfts[idx], itrs[idx], "(" + typ[idx] + ")"
    ec_("Z7:  before return;", dbf > 0)
    if full:
        return retval
    return retval[0], retval[1]


def prm_worker(poly, delta_method, prms, nmsp):
    """Call prm_mult and prm_sing, put mlt, msft, itr, zout on namespace."""
    dbf = 0
    ec_("ZZ1", dbf > 0)
    nmsp.zout, nmsp.mlt, nmsp.msft, nmsp.itr = prms(poly, delta_method)
    ec_("ZZ2", dbf > 0)
    return


def apprtsat0(zout, mlt, rtsat0):
    """Add root at zero and its multiplicity to non-zero roots found."""
    idx = np.where(mlt != 0)
    zout[idx[0][-1] + 1] = ZERO
    mlt[idx[0][-1] + 1] = rtsat0
    return zout, mlt


def killprocs(proc):
    """Kill all background processes."""
    dbf = 0
    for j, _ in enumerate(proc):
        fmt = "A4 before proc[%d].is_alive()=%d;"
        ec_(fmt % (j, proc[j].is_alive()), dbf > 0)
        if proc[j].is_alive():
            proc[j].terminate()
        fmt = "A5 after proc[%d].is_alive()=%d;"
        ec_(fmt % (j, proc[j].is_alive()), dbf > 0)


def prm_mult(poly, delta_method):
    """Find roots of a polynomial using method better for multiplicities > 1.

    Calculate roots of polynomial poly, finding multiplicities using roots
    of the derivatives.  Recursive.

    Parameters
    ----------
    poly : list or array like
        coefficients of the polynomial -- size n+1
        integer, float, or multiple precision, real or complex
    delta_method : integer
        delta method to be used -- see Notes

    y(x) = poly[0]*x^n + poly[1]*x^(n-1) + ... + poly[n-1]*x + poly[n]

    Returns
    -------
    zout : ndarray
        multiple precision ndarray containing the roots of the polynomial
    mlt : list
        multiplicity of zout
    msf : float
        figure of merit of root estimates -- larger negative numbers are better
        log10(sum(abs(poly(zout))) / n)) / 2
    num_iter : int
        total number of iterations

    Notes
    -----
    delta_method
        0    delta_weierstrass  WeierStrass-Durand-Kerner
        1    delta_aberth       Aberth-Ehrlich
        3    delta_sakurai      Sakurai-Torii-Sugiura
        7    delta_petkovic     Sakurai-Petkovic

    Example
    -------
    >>> mp.mp.dps = 2000
    >>> prm_mult([1, 4, 6, 4, 1], 1)
    (array([mpc(real='-1.0', imag='0.0'), mpc(real='0.0', imag='0.0'),
           mpc(real='0.0', imag='0.0'), mpc(real='0.0', imag='0.0')],
           dtype=object), [4, 0, 0, 0], -4000, 4)
    """

    cst = time.time()
    roottol = mp.mpf(1e-200)
    dbf = 0                 # -1:4   1
    msft = 0
    itr = 1
    num_iterd = 0
    # -------------------------------------------------------------------------
    poly, _ = polytrim(poly)

    npoly = len(poly) - 1
    ec_(FM01 % (npoly), dbf > 2)
    if npoly >= 2:          # derivatives
        polyd = np.polyder(poly)

        cst = time.time() - cst
        znd, mltd, _, num_iterd = prm_mult(polyd, delta_method)           # ---
        cst = time.time() - cst

        if dbf > 2:
            ec_("AA5:  n=%d;  polyd, znd, mltd:" % (npoly))
            for idx in np.where(mltd)[0]:
                print(" %d" % mltd[idx], end="")
            print(flush=True)
        prntall(polyd, znd, mltd, dbf > 4)

    if npoly == 0:          # only roots at zero
        zout = [mp.mpf('0')]
        mlt = [0]
        ec_("AA3", dbf > 2)
    elif npoly == 1:        # line        0 = poly[0]*x + poly[1]
        zout = [-poly[1]]
        mlt = [1]
        msft = msemsf(poly, zout, mlt)
        ec_("AA4", dbf > 2)
    elif npoly == 2:        # quadratic   0 = poly[0]*x^2 + poly[1]*s + poly[2]
        if mltd[0] == 1 and mp.fabs(mppv(poly, znd[0])) < roottol:
            zout = [znd[0], ZERO]
            mlt = [2, 0]
        else:
            sym = poly[1] / TWO
            rad = mp.sqrt(sym * sym - poly[2])
            zout = [-sym + rad, -sym - rad]
            mlt = [1, 1]
        msft = msemsf(poly, zout, mlt)
    else:                   # npoly >= 3;  cubic and higher
        ec_("BB1", dbf > 2)
        zeroes = mpmpc([0] * npoly)
        znew = np.copy(zeroes)
        zout = np.copy(zeroes)
        mlt = [0] * npoly

        # check znd[:] as roots and update multiplicity
        idz = 0
        ide = 0
        rava = np.copy(zeroes)
        mava = np.zeros(npoly, dtype=int)
        ec_("BB2", dbf > 2)
        for idx in np.where(mltd)[0]:
            if mp.fabs(mppv(poly, znd[idx])) < roottol:
                znew[idz] = znd[idx]
                mlt[idz] = mltd[idx] + 1
                idz += 1
            else:           # collect derivative roots for use in estimate
                rava[ide] = znd[idx]
                mava[ide] = mltd[idx]
                ide += 1
        # find roots of poly knowing {znew,mlt}[0:idz]
        # starting points are modifications of roots of derivative
        nrootsm = sum(mlt)      # # roots found from mulitiplicities
        nrootsq = npoly - nrootsm   # # roots still needed
        nrootsd = sum(mava)     # # roots avilable from derivative
        fmt = "BB2a: n=%d; idz=%d; ide=%d; nrootsm=%d; nrootsq=%d; nrootsd=%d;"
        ec_(fmt % (npoly, idz, ide, nrootsm, nrootsq, nrootsd), dbf > 0)
        if nrootsq == 0:        # all roots found from multiplicity
            ec_("BB3:  n=%d; nrootsq == 0" % (npoly), dbf > 3)        # test 3
            return znew, mlt, msemsf(poly, znew, mlt), itr + num_iterd
        if nrootsd == 0:        # no roots from derivative:  poly(x) = x^n +c
            # polyn = [2, 3, 4, 8, 9, 81, 82, 83, 84, 85, 86, 89]
            mag = mp.exp(mp.log(mp.fabs(poly[npoly])) / nrootsq)
            ang = (2 * mp.pi * np.array(mp.arange(nrootsq)) +
                   mp.arg(ZERO - poly[npoly])) / mp.mpf(nrootsq)
            for jjj in range(idz, (idz + nrootsq)):
                mlt[jjj] = 1
                znew[jjj] = mag * (mp.cos(ang[jjj - idz]) +
                                   mp.sin(ang[jjj - idz]) * 1J)
            fmt = "BB4:  n=%d; mag=%20.12e; msemsf=%9.2f;"
            ec_(fmt % (npoly, mag, msemsf(poly, znew, mlt)), dbf > 0)  # test 2
        else:
            iava = 0
            nava = sum(mava > 0)
            for idw in range(idz, idz + nrootsq):
                mlt[idw] = 1
                rndx = (1 + (1 + 1j) / 1000.) ** (idw - idz + 1)
                if sum(mava) > 0:
                    while mava[iava] == 0:
                        iava = (iava + 1) % nava
                    znew[idw] = rava[iava] * rndx
                    mava[iava] -= 1
                else:
                    znew[idw] = rava[iava] * rndx
                    iava = (iava + 1) % nava
            ec_("BB5:  znew estimate", dbf > 2)
            ec_(FM11 % (npoly), dbf > 0)                          # test 2
            prntz(znew, mlt, dbf > 0)                             # test 2

        zout, _, itr = mpr(poly, polyd, znew, mlt, dbf, delta_method)     # ---

        ec_("KK", dbf > 2)
        msft = msemsf(poly, zout, mlt)
        cst = time.time() - cst
        ec_(FM39 % (npoly, delta_method, itr, mp.mp.dps, msft, cst), dbf > 0)
        # npoly >= 3:
    prntz(zout, mlt, dbf > 0)       # 1
    ec_("XX", dbf > 0)
    return zout, mlt, msft, itr + num_iterd


def prm_sing(poly, delta_method):
    """Find roots of a polynomial using method better for multiplicities == 1.

    Parameters
    ----------
    poly : list or array like
        coefficients of the polynomial -- size n+1
        integer, float, or multiple precision, real or complex
    delta_method : integer
        delta method to be used -- see Notes

    y(x) = poly[0]*x^n + poly[1]*x^(n-1) + ... + poly[n-1]*x + poly[n]

    Returns
    -------
    zout : ndarray
        multiple precision ndarray containing the roots of the polynomial
    mlt : list
        multiplicity of zout
    msf : float
        figure of merit of root estimates -- larger negative numbers are better
        log10(sum(abs(poly(zout))) / n)) / 2
    num_iter : int
        total number of iterations

    Notes
    -----
    delta_method
        0    delta_weierstrass  WeierStrass-Durand-Kerner
        1    delta_aberth       Aberth-Ehrlich
        3    delta_sakurai      Sakurai-Torii-Sugiura
        7    delta_petkovic     Sakurai-Petkovic

    Example
    -------
    >>> mp.mp.dps = 20
    >>> prm_sing([1, -10, 35, -50, 24], 3)
    (array([mpc(real='2.000000000000000000044', imag='-9.629649721936179e-34'),
           mpc(real='1.0000000000000000000017', imag='6.018531076210112e-36'),
           mpc(real='2.9999999999999999999492', imag='-3.37037740267766e-33'),
           mpc(real='4.000000000000000000061', imag='-3.291384182302405e-36')],
           dtype=object), [1, 1, 1, 1], mpf('-18.582708243881810972198'), 7)
    """

    cst = time.time()
    dbf = 0                 # -1:4   1

    poly, _ = polytrim(poly)
    npoly = len(poly) - 1
    ec_(FM01 % (npoly), dbf > 2)
    polyd = np.polyder(poly)
    znew = rootmax11(poly)                                                # ---
    ec_("BB5:  znew estimate", dbf > 2)
    ec_(FM11 % (npoly), dbf > 0)                                  # test 2
    mlt = [1] * npoly
    zout, _, itr = mpr(poly, polyd, znew, mlt, dbf, delta_method)         # ---
    ec_("KK", dbf > 2)
    msft = msemsf(poly, zout, mlt)
    cst = time.time() - cst
    ec_(FM39 % (npoly, delta_method, itr, mp.mp.dps, msft, cst), dbf > 0)
    ec_("WW", dbf > 0)
    return zout, mlt, msft, itr


def polytrim(poly):
    """Remove leading and trailing coefficients of zero."""
    dbf = 0
    ec_("AA1i", dbf > 0)
    polyc = mpmpc(poly)
    # Exception: STATUS_ACCESS_VIOLATION:  np.where needs comparison n ~ 600
    idx = np.where(polyc != ZERO)    # where returns indeces of non-zero values
    ec_("AA1j", dbf > 0)
    polyr = polyc[idx[0][0]:(idx[0][-1] + 1)]   # remove leading and trailing
    if polyc[0] != ONE:
        polyr = polyr / polyr[0]                # normalize
    ec_("AA1o", dbf > 0)
    return polyr, len(polyc) - idx[0][-1] - 1   # return poly and # trailing 0s


def mpr(poly, polyd, znew, mlt, dbf, delta_method):
    """Find roots of a polynomial given starting points and multiplicities.

        called by prm_mult, prm_sing
        calls delta_weierstrass, delta_aberth, delta_sakurai, delta_petkovic
        method agnostic
    """

    npoly = len(poly) - 1
    zeroes = mpmpc([0] * npoly)
    itera = 1000
    change_tol = mp.mpf(10) ** (-mp.mp.dps // 2)
    msf = np.ones(itera) * 10000
    polydd = np.polyder(polyd)
    brkflg = 0
    itr = 0
    for itr in np.arange(1, itera):
        zold = np.copy(znew)
        change = 0
        fmt = "CC:   n=%d; itr=%d; digits=%d;"
        ec_(fmt % (npoly, itr, mp.mp.dps), dbf > 2)
        delta = np.copy(zeroes)
        zoabs = np.abs(zold[:])
        for jji in np.arange(npoly):
            ec_("DD:   n=%d; itr=%d; jji=%d;" % (npoly, itr, jji), dbf > 3)
            if mlt[jji] != 1:
                continue
            ec_("CC2:  n=%d; itr=%d; jji=%d;" % (npoly, itr, jji), dbf > 3)
            val = mppv(poly, zold[jji])
            ec_("CC3", dbf > 3)
            if delta_method == 0:
                ec_("CC4", dbf > 3)
                delta[jji] = delta_weierstrass(poly, zold, zoabs, val, mlt,
                                               jji, itr, dbf)             # ---
            if delta_method == 1:
                vald = mppv(polyd, zold[jji])
                ec_("CC4", dbf > 3)
                delta[jji] = delta_aberth(poly, zold, zoabs, val, vald, mlt,
                                          jji, itr, dbf)                  # ---
            if delta_method == 3:
                vald = mppv(polyd, zold[jji])
                ec_("CC4", dbf > 3)
                valdd = mppv(polydd, zold[jji])
                ec_("CC5", dbf > 3)
                delta[jji] = delta_sakurai(poly, zold, zoabs, val, vald,
                                           valdd, mlt, jji, itr, dbf)     # ---
            if delta_method == 7:
                vald = mppv(polyd, zold[jji])
                ec_("CC4", dbf > 3)
                valdd = mppv(polydd, zold[jji])
                ec_("CC5", dbf > 3)
                delta[jji] = delta_petkovic(poly, zold, zoabs, val, vald,
                                            valdd, mlt, jji, itr, dbf)    # ---
            fmt = "FF9:  delta[%d]=%+20.12e %+20.12e j; "
            ec_(fmt % (jji, delta[jji].real, delta[jji].imag), dbf > 3)

            ec_("FF10", dbf > 3)
            dvt = np.abs(znew[jji]) + zoabs[jji]
            if dvt == ZERO:
                ec_("np.abs(znew[jji])+np.abs(zold[jji]) is zero")
                return
            dtxv = np.abs(delta[jji]) / dvt
            fmt = "FF11:  n=%d; itr=%3d; jji=%3d; dvt=%15.9e"
            ec_(fmt % (npoly, itr, jji, dvt), dbf > 3)
            if itr == 2 or change < dtxv:
                change = dtxv
            ec_("FF12", dbf > 3)
            znew[jji] = zold[jji] + delta[jji]

            znjj = np.complex(znew[jji])
            fmt = "GG2c:   znjj=%+15.8e %+15.8e j;"
            ec_(fmt % (np.real(znjj), np.imag(znjj)), dbf > 2)
            # jji
        msf[itr] = msemsf(poly, znew, mlt)

        if dbf > 1:
            fmt = "HH1:  n=%d; change[%2d,%3d]=%9.2e; msf=%9.2f; digits=%d; II"
            fmu = "HH2:  n=%d; log10(change[%2d,%3d])=%9.2f; " + \
                "msf=%9.2f; digits=%d; II"
            if np.abs(change) == ZERO:
                ec_(fmt % (npoly, delta_method, itr, np.abs(change), msf[itr],
                           mp.mp.dps))
            else:
                ec_(fmu % (npoly, delta_method, itr, mp.log10(np.abs(change)),
                           msf[itr], mp.mp.dps))
        if change < change_tol or msf[itr] < -mp.mp.dps * 0.95:
            zout = np.copy(znew)
            brkflg = 1
            break
        # itr
        if brkflg == 0:
            zout = np.copy(znew)
    return zout, msf, itr


def delta_weierstrass(poly, zold, zoabs, val, mlt, jji, itr, dbf):
    """Calculate delta using WeierStrass-Durand-Kerner.

    Reference:
        https://en.wikipedia.org/wiki/Durand%E2%80%93Kerner_method
    """

    npoly = len(poly) - 1
    acc = ONE
    for kki in np.arange(npoly):
        if mlt[kki] == 0 or jji == kki:
            continue
        ec_(FM18 % (npoly, itr, jji, kki), dbf > 3)
        if roottoltest(zold, zoabs, jji, kki):
            acc *= (zold[jji] - zold[kki]) ** mlt[kki]
    ec_("FF1", dbf > 3)
    if acc == ZERO:
        ec_("FF6:  acc is zero")
        return
    ec_("FF7:  acc is one", (mp.fabs(acc - 1) < mp.mpf(0.01)))
    delta = -val / acc
    ec_(FM28 % (jji, val.real, val.imag, acc.real, acc.imag), dbf > 3)
    return delta


def delta_aberth(poly, zold, zoabs, val, vald, mlt, jji, itr, dbf):
    """Calculate delta using Aberth-Ehrlich.

    Reference:
        https://en.wikipedia.org/wiki/Aberth_method
    """

    npoly = len(poly) - 1
    acc = ZERO
    for kki in np.arange(npoly):
        if mlt[kki] == 0 or jji == kki:
            continue
        ec_(FM18 % (npoly, itr, jji, kki), dbf > 3)
        if roottoltest(zold, zoabs, jji, kki):
            dvt = zold[jji] - zold[kki]
            if dvt == ZERO:
                ec_("zold[delta_method,jji]-zold[delta_method,kki] is zero")
                return
            one_dvx = mp.mpf(mlt[kki]) / dvt
            acc += one_dvx
    ec_("FF1", dbf > 3)
    dvt = vald - val * acc
    if dvt == ZERO:
        ec_("FF2:  vald[jji]-val[jji] * acc is zero")
        dvt = ONE
    delta = -val * ONE / dvt
    ec_(FM28 % (jji, val.real, val.imag, acc.real, acc.imag), dbf > 3)
    return delta


def delta_sakurai(poly, zold, zoabs, val, vald, valdd, mlt, jji, itr, dbf):
    """Calculate delta using Sakurai-Torii-Sugiura.

    Reference:
        T. Sakurai, M.S. Petkovic  1996-Aug-13
        On some simultaneous methods based on Weierstrass correction
        Section 2, Sakurai-Torii-Sugiura
    """

    del1, del2, sum1, sum2 = dels_sums(poly, zold, zoabs, val, vald, valdd,
                                       mlt, jji, itr, dbf)                # ---
    ec_("FF1", dbf > 3)
    dvt = del2 + TWO * (sum1 * del1 - del1 * del1) + sum2 - sum1 * sum1
    if dvt == ZERO:
        delta = ZERO
    else:
        delta = -TWO * (sum1 - del1) / dvt
    ec_(FM28 % (jji, val.real, val.imag, sum1.real, sum1.imag), dbf > 3)
    return delta


def delta_petkovic(poly, zold, zoabs, val, vald, valdd, mlt, jji, itr, dbf):
    """Calculate delta using Sakurai-Petkovic.

    Reference:
        T. Sakurai, M.S. Petkovic  1996-Aug-13
        On some simultaneous methods based on Weierstrass correction
        Section 2, Sakurai-Petkovic
    """

    del1, del2, sum1, sum2 = dels_sums(poly, zold, zoabs, val, vald, valdd,
                                       mlt, jji, itr, dbf)                # ---
    dvt = TWO * (del1 - sum1) ** THREE
    if dvt == ZERO:
        delta = ZERO
    else:
        delta = (-(THREE * (del1 - sum1) ** TWO + del2 - del1 * del1 + sum2) /
                 dvt)
    ec_(FM28 % (jji, val.real, val.imag, sum1.real, sum1.imag), dbf > 3)
    return delta


def dels_sums(poly, zold, zoabs, val, vald, valdd, mlt, jji, itr, dbf):
    """Calculate the dels and sums used in the various delta methods."""
    npoly = len(poly) - 1
    sum1 = ZERO
    sum2 = ZERO
    del1 = ZERO
    del2 = ZERO
    if val != ZERO:
        del1 = vald / val
        del2 = valdd / val
    for kki in np.arange(npoly):
        if mlt[kki] == 0 or jji == kki:
            continue
        ec_(FM18 % (npoly, itr, jji, kki), dbf > 3)
        if roottoltest(zold, zoabs, jji, kki):
            dvt = zold[jji] - zold[kki]
            if dvt == ZERO:
                ec_("Zo[ab,jj]-Zo[ab,kk] is zero")
                return
            # will this work for delta_method == 3?
            one_dvx = mp.mpf(mlt[kki]) / dvt
            sum1 += one_dvx
            sum2 += one_dvx * one_dvx
    return del1, del2, sum1, sum2


def rootmax11(poly):
    """Calculate root estimates using Krishnan-Foskey-Culver-Keyser-Manocha.

    Reference:
        Krishnan, Foskey, Culver, Keyser, Manocha  2001-Jun-03
        PRECISE:  Efficent Multiprecision Evaluation of Algebraic Roots and
        Predicates for Reliable Geometric Computation
        Section 4.2 Choice of Initial Approximations
    """

    polya = np.abs(poly)
    npoly = np.size(poly) - 1
    ppolya = polya[1:]
    rmax = min(max(ONE, np.sum(ppolya)), ONE + np.max(ppolya))
    ppolya = polya[:-1]
    rmin = polya[-1] / min(max(polya[-1], np.sum(ppolya)),
                           (polya[-1] + np.max(ppolya)))

    srng = mpmpc([0] * (npoly + 1))
    rrng = np.copy(srng)
    rrng[0] = rmin
    srng[-1] = rmax
    kkm = 0
    k_list = np.zeros(npoly + 1)
    skk = np.copy(srng)
    rkk = np.copy(srng)
    for kki in [0, npoly] + list(range(1, npoly)):
        pak = np.copy(polya)
        pak[npoly - kki] = 0 - polya[npoly - kki]
        if kki == 0:
            srng[kki] = fndlim(pak, rrng[kki], -1, rmax)                  # ---
            rmin = max(rmin, srng[kki])
            k_list[kkm] = kki
            skk[kkm] = srng[kki]
            kkm += 1
        elif kki == npoly:
            rrng[kki] = fndlim(pak, rmin, 1, rmax)                        # ---
            rmax = min(rmax, rrng[kki])
        else:
            rrng[kki] = fndlim(pak, skk[kkm - 1], 1, rmax)                # ---
            srng[kki] = ZERO
            if rrng[kki] <= rmax:
                srng[kki] = fndlim(pak, rrng[kki], -1, rmax)              # ---
            if (skk[kkm - 1] <= rrng[kki] and rrng[kki] <= srng[kki] and
                    srng[kki] <= rmax):
                k_list[kkm] = kki
                skk[kkm] = srng[kki]
                rkk[kkm] = rrng[kki]
                kkm += 1
    k_list[kkm] = npoly
    rkk[kkm] = rrng[npoly]

    jjj = 0
    znew = mpmpc([0] * npoly)
    for mmi in np.arange(kkm):
        nnn = k_list[mmi + 1] - k_list[mmi]
        phi = mp.pi / (mp.mpf(2) * nnn)
        for lli in np.arange(1, nnn + 1):
            radius = rkk[mmi + 1]
            znew[jjj] = radius * mp.exp(1j * (2 * mp.pi * lli / nnn + phi))
            jjj += 1
    return znew


def fndlim(pak, rsk, sgn, rmax):
    """Find limits in rootmax11."""
    rskk = rsk
    fact = TWO
    fact_cnt_max = 3
    fexp = TWO
    fact_cnt = 0
    factp = fact

    val0 = mppv(pak, rskk)
    while val0 * sgn > 0 and rskk <= rmax:
        rskk = rskk * factp
        val0 = mppv(pak, rskk)
        if val0 * sgn < 0 and fact_cnt < fact_cnt_max:
            rskk = rskk / factp
            val0 = mppv(pak, rskk)
            factp = factp ** (1 / fexp)
            fact_cnt += 1
    if sgn == -1:
        rskk = rskk / factp
    return rskk


def ec_(msg="", doit=True, timestr=True):
    """Print with timestamp."""
    if doit:
        fnam = extract_stack(None, 2)[0][2]     # function calling ec_
        ec_time = ""
        if timestr:
            ec_time = time.strftime("%Y-%b-%d %H:%M:%S ")
        print('[%s%s] %s' % (ec_time, fnam, msg), flush=True)


def mppv(poly, znew):
    """Calculate value of polynomial."""
    return mp.polyval(poly.tolist(), znew)


def prod(factors):
    """Calculate the product of the factors."""
    return reduce(mul, factors, 1)


def prnt(strng, doit=True):
    """Print if."""
    if doit:
        print(strng, end="", flush=True)


def hostn(timestr=True):
    """Print the hostname, number of cpus and the version using ec_."""
    fmt = "host=%s; cpu=%d; ver=%s"
    host = gethostname()
    cpus = cpu_count()
    vers = sys.version.replace('\n', ' ')
    ec_(fmt % (host, cpus, vers), True, timestr)
    return


def roottoltest(zold, zoabs, jji, kki):
    """Check difference between roots."""
    root_diff_tol = mp.mpf(10) ** (-mp.mp.dps * 10)
    return (mp.fabs(zold[jji] - zold[kki]) > root_diff_tol *
            (zoabs[jji] + zoabs[kki]))


def msemsf(poly, znew, mlt):
    """Calculate sum of polynomial values at root estimates."""
    valasum = ZERO
    npoly = np.size(poly) - 1
    for idx in np.where(mlt)[0]:
        vala = mp.fabs(mppv(poly, znew[idx]))
        valasum += vala * vala * mlt[idx]
    if valasum == ZERO:
        msf = -mp.mp.dps * 2
    else:
        msf = np.float(mp.log10(valasum / mp.mpf(npoly))) / TWO
    return msf


def prntall(poly, znew, mlt, doit=True):
    """Print polynomial coefficients, roots and multiplicities."""
    if not doit:
        return
    fmt = "poly[%d]=%+20.12e %+20.12e j;"
    for idx, pidx in enumerate(poly):
        ec_(fmt % (idx, pidx.real, pidx.imag))
    prntz(znew, mlt)
    return


def prntz(znew, mlt, doit=True, timestr=True):
    """Print roots and multiplicities."""
    if not doit:
        return
    fmt = "n=%d; zn[%2d]=%+20.12e %+20.12e j;  mlt=%d;"
    npoly = len(znew)
    idz = np.argsort(np.array(znew, dtype=complex))
    for idx in range(npoly):
        idy = idz[idx]
        if mlt[idy] == 0:
            continue
        ec_(fmt % (npoly, idy, znew[idy].real, znew[idy].imag, mlt[idy]), True,
            timestr)
    return


def mpmpc(inlist):
    """Convert list of numbers into array of multiple precision complex."""
    outlist = np.array(mp.zeros(len(inlist), 1))
    for idx, adx in enumerate(inlist):
        adxm = adx
        if isinstance(adx, np.int64):
            adxm = int(adx)
        outlist[idx] = mp.mpc(adxm)
    return outlist


def polysets(ntest=False):
    """generate list of tests"""
    x00 = list(range(-1, -15, -1))                      # simple ones for prm
    y00 = list(range(1, 18))
    # y01 = list(range(16, 18))

    u00 = list(range(24, 31))                           # multiple roots
    # u01 = list(range(29, 31))                           # multiple roots
    q00 = [32, 35, 40]                                  # multiple roots
    t00 = [45, 50, 60, 70, 80]                          # multiple roots

    # The v00 set can take long, and 89 and 90 can take even longer.
    v00 = [81, 82, 83, 84, 85, 86, 88, 91, 92, 93, 89, 90]  # 87
    # v01 = list(range(81, 87)) + list(range(88, 94))
    # v04 = list(range(81, 87))
    # v02 = list(range(90, 94))
    # v03 = list(range(92, 94))
    w00 = [110, 120, 150, 195, 196, 197, 198, 199]      # 197, 198: errors
    # Exception: STATUS_ACCESS_VIOLATION at rip=003961FAB36

    ct0 = list(range(203, 212))                         # Chebyshev T
    cu0 = list(range(303, 310))                         # Chebyshev U

    polyset = [400, 401, 402] * 10
    polyset = w00
    polyset = [-7, -8, -13, -14]                        # 2 quintics mult & non
    polyset = x00 + y00 + u00 + q00 + t00 + ct0 + cu0 + v00  # standard test

    nset = [None]
    if ntest:                                           # test at multiple n
        nset = [3, 4, 5, 6, 8, 10,
                13, 17, 22, 28, 36, 46, 60, 77, 100,
                130, 170, 220, 280, 360, 460, 600, 770, 1000,
                1300, 1700, 2200, 2800, 3600, 4600, 6000, 7700, 9999]

        polyset = [1, 5, 6, 7, 9, 12, 17, 85, 86, 91, 92, 93]  # 87
        polyset = [17, 9, 85, 1, 86, 5, 12, 92, 6, 93, 91, 7]  # sorted (n=100)
        polyset = [9, 17, 85, 1, 86, 5, 12, 92, 6, 93, 91, 7]  # 17 stuck  1000
    return polyset, nset


def prm_test():
    """Perform test of prm using a set of polynomials."""
    dbf = 1
    mp.mp.dps = 1000
    mp.mp.dps = 10000
    mp.mp.dps = 2000

    fme = "=================================================================\n"
    fmd = "-----------------------------------------------------------------\n"

    polyset, nset = polysets()
    hostn(False)
    for polyn in polyset:
        for npol in nset:
            prnt(fme, dbf > 0)
            cpg = time.time()
            poly, npoly, mls = polygen(polyn, npol)
            cpg = time.time() - cpg
            if dbf > 1:
                znn = rootmax11(poly)
                zvv = np.sort(np.array(znn, dtype=complex))
                for idx in np.arange(npoly):
                    fmt = 'zvv[%2d] = %+20.12e %+20.12e j;'
                    ec_(fmt % (idx, zvv[idx].real, zvv[idx].imag))
                prnt(fmd)
            # continue                            # uncomment for rootmax only

            cst = time.time()
            (znew, mlt, msf, num_iter, tstr) = prm(poly, delta_method=[1, 3],
                                                   full=True)
            cst = time.time() - cst
            mltf = "no"
            for jji in np.arange(len(mlt)):
                if mlt[jji] > 1:
                    mltf = "yes"

            fmt = 'prm=%3i; n=%4i; it=%4i; dig=%5i; msf=%9.2f; gen=%8.2f; ' + \
                'sec=%8.2f; sec/it=%8.2f; %3s/%3s/%4s;'
            print(fmt %
                  (polyn, npoly, num_iter, mp.mp.dps, msf, cpg, cst,
                   cst / num_iter, mls, mltf, tstr), flush=True)
            prntz(znew, mlt, dbf > 0, False)
            prnt(fmd, dbf > 0)
            print_poly = False
            if print_poly:
                for idx, pidx in enumerate(poly):
                    fmt = 'p[%d] = %+20.12e %+20.12e j; '
                    ec_(fmt % (idx, pidx.real, pidx.imag))

    print()


def polygen(polyn=-1, npoly=None):
    """Generate a test polynomial."""
    J = 1j * ONE
    four = mp.mpf(4)
    six = mp.mpf(6)
    nine = mp.mpf(9)
    mls = "no"

    if polyn == -1:         # line
        poly = mpmpc([3, 6])
    elif polyn == -2:       # parabola
        poly = mpmpc([1, -2, 3])
    elif polyn == -3:       # cubic
        poly = mpmpc([1] * 4)
    elif polyn == -4:       # cubic                         # mult
        poly = mpmpc([1, 3, 3, 1])
        mls = "yes"
    elif polyn == -5:
        poly = mpmpc([1] * 5)
    elif polyn == -6:                                       # mult
        poly = mpmpc([1, 4, 6, 4, 1])
        mls = "yes"
    elif polyn == -7:
        poly = mpmpc([1] * 6)
    elif polyn == -8:                                       # mult
        poly = mpmpc([1, 5, 10, 10, 5, 1])
        mls = "yes"
    elif polyn == -9:   # np.poly([1, 2, 2, 3, 3, 3, 4])    # mult
        poly = mpmpc([1, -18, 136, -558, 1339, -1872, 1404, -432])
        mls = "yes"
    elif polyn == -10:  # np.poly([1, 2, 2])                # mult
        poly = mpmpc([1, -5, 8, -4])
        mls = "yes"
    elif polyn == -11:  # np.poly([1, 2, 2, 2])             # mult
        poly = mpmpc([1, -7, 18, -20, 8])
        mls = "yes"
    elif polyn == -12:  # np.poly([1, 2, 2, 5, 5])          # mult
        poly = mpmpc([1, -15, 83, -209, 240, -100])
        mls = "yes"
    elif polyn == -13:                                      # roots at 0
        poly = mpmpc([0] * 2 + [1] * 6 + [0] * 3)
    elif polyn == -14:                                      # mult, roots at 0
        poly = mpmpc([0, 0, 1, 5, 10, 10, 5, 1, 0, 0, 0])
        mls = "yes"
    elif polyn == -15:
        # https://math.stackexchange.com/questions/2361040/
        # influence-of-small-constant-term-on-roots-of-polynomial
        poly = mpmpc([mp.mpf(-2144171792184977) / mp.mpf(36028797018963968),
                      mp.mpf(-3320294798501755) / mp.mpf(9007199254740992),
                      mp.mpf(-56940771119885) / mp.mpf(70368744177664),
                      mp.mpf(-187473016346583) / mp.mpf(281474976710656),
                      mp.mpf(-152325066937821) / mp.mpf(18014398509481984),
                      mp.mpf(1520316102106201) / mp.mpf(9007199254740992),
                      mp.mpf(1) / mp.mpf(36028797018963968)])
        mls = "no"

    elif polyn == 1:
        if npoly is None:
            npoly = 30
        poly = mpmpc([1] * (npoly + 1))
        for kki in np.arange(1, npoly + 1):
            poly[npoly - kki] = poly[npoly - kki + 1] * kki
    elif polyn == 2:                                        # mult ?
        cnst = mp.mpf(1e20)
        poly = mpmpc([J / cnst, 0, 0, 0, 0, 1, -6 / (cnst * cnst),
                      9 / (cnst * cnst * cnst * cnst)])
        mls = "may"
    elif polyn == 3:
        cnst = mp.mpf(1e20)
        poly = mpmpc([cnst * cnst, 0, 0, 0, 0, cnst * cnst * cnst * cnst, 0,
                      -6 * cnst * cnst, 0, 9])
    elif polyn == 4:
        cnst = 1e12
        poly = mpmpc([1, 0, 0, 0, 0, 0, cnst, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                      0, 0, 1])
# http://www-sop.inria.fr/saga/POL/BASE/1.unipol/wilkinson.1.n.dir/index.html
    elif polyn == 5:                    # Wilkinson's Polynomial
        if npoly is None:
            npoly = 30                  # standard
        # npoly = 40                    # takes much longer than 20
        # npoly = 20
        # npoly = 200
            #  40 ok with maxloop=1e5                   (4.9m laptop cygwin)
            #  60 ok with maxloop=1e5                   (16m laptop cygwin)
            # 100 ok with maxloop=1e5                   (1.3h laptop cygwin)
            # 150 fails with maxloop=1e5, ok with 1e6   (7.6h laptop cygwin)
            # 200 ok with 1e6                           (12h laptop cygwin)
        zav = np.array(mp.arange(1, npoly + 1))
        poly = np.poly(zav)
# Wilkinson's perturbation from
# https://en.wikipedia.org/wiki/Wilkinson's_polynomial
    # poly[1] = poly[1] - TWO ** (mp.mpf(-23))
    elif polyn == 6:
        if npoly is None:
            npoly = 20
        p00 = mpmpc([0.01, 0, 0, 0, 0, 0, 0, 0, 1, -20, 100])
        zav = np.array(mp.arange(1, npoly + 1))
        p01 = np.poly(zav)
        poly = np.polymul(p01, p00)
    elif polyn == 7:                                        # mult
        if npoly is None:
            npoly = 5
        p01 = np.array(([ONE, mp.mpf(-40), mp.mpf(400)]))
        poly = ONE
        p00 = mpmpc([1] * 2)
        for kki in np.arange(1, npoly + 1):
            p00[1] = ONE * kki
            poly = np.polymul(poly, p00)
            poly = np.polymul(poly, p01)
        mls = "yes"
    elif polyn == 8:
        cnst = mp.mpf(1e24)
        poly = mpmpc([1, 0, 0, 2 * cnst, 0, 0, cnst * cnst, 4, 0, 0, -4 * cnst,
                      0, 0, 0, 4])
# http://www-sop.inria.fr/saga/POL/BASE/1.unipol/nthroot.1.n.dir/index.html
    elif polyn == 9:                    # N-th Roots
        if npoly is None:
            npoly = 40
        # npoly = 2000
        # npoly = 200
        cnst = mp.mpf(37)
        poly = mpmpc([0] * (npoly + 1))
        poly[0] = ONE                   # n
        poly[npoly] = cnst              # 0
    elif polyn == 10:  # notorious example poly(z) = z ** 4 - z ** 2 - 11 / 36
        poly = mpmpc([1, 0, -1, 0, -11 / 36])
    elif polyn == 11:
        # poly(x) = x ** 3 - 2 * x + 2;  # super attracting 0 -> 1 -> 0
        poly = mpmpc([1, 0, -2, 2])
    elif polyn == 12:
        # causes a seg fault (core dumped) in aberth when 1:6 are 10
        # does not converge
        if npoly is None:
            npoly = 21
        zav = mpmpc([0] * npoly)
        poly = mpmpc([0] * (npoly + 1))
        nni = npoly - 1
        incr = 0.1
        for kki in np.arange(nni):
            zav[kki] = mp.mpf(10 + (kki + 1) * incr)
        zav[nni] = mp.mpf("9")
        poly = np.poly(zav)
    elif polyn == 13:                                       # mult
        # example 1 from "On and Efficient Method for the Simultaneous
        # Approximation of Poynomial Multiple Roots"
        # 3 x 2, 5 x +i,5 x -i
        # (x - 2) ** 3 * (x ** 2 + 1) ** 5
        # (x ** 3 - 6 * x ** 2 + 12 * x - 8) * (x ** 10 + 5 * x ** 8 + 10 *
        # x ** 6 + 10 * x ** 4 + 5 * x ** 2 + 1)
        poly = mpmpc([1, -6, 17, -38, 70, -100, 130, -140, 125, -110, 61, -46,
                      12, -8])
        mls = "yes"
    elif polyn == 14:                                       # mult
        # example 2 from "On and Efficient Method for the Simultaneous
        # Approximation of Poynomial Multiple Roots"
        # 2 x -1, 3 x -3, 2 x 1+i, 2 x 1-i, 3 x 1, 2 x +2+i, 2 x +2-i,
        #   2 x -2+i, 2 x -2-i
        poly = mpmpc([1, 4, -20, -72, 252, 664, -2092, -3440, 12450, 9520,
                      -51476, -1264, 142360, -82488, -228612, 279376, 117237,
                      -337300, 77400, 135000, -67500])
        mls = "yes"
    elif polyn == 15:                                       # mult
        # example 2 from "On and Efficient Method for the Simultaneous
        # Approximation of Poynomial Multiple Roots"
        # 2 x -1, 3 x -2, 3 x +2, 2 x 1+i, 2 x 1-i, 2 x +i, 2 x -i, 2 x -2+i
        poly = mpmpc([1, 2 - J * 2, -14, -18 + J * 26, 80 + J * -12,
                      26 + J * -118, -238 + J * 136, 146 + J * 182,
                      307 + J * -476, -380 + J * 160, 236 + J * 320,
                      32 + J * -712, -804 + J * 880, 512 + J * 96,
                      -80 + J * -832, -1024 + J * 1152, -448 + J * 256,
                      -1024 + J * 512, -768 + J * 1024])
        mls = "yes"
    elif polyn == 16:                                       # mult
        # 5 x +i, 5 x -i
        poly = mpmpc([1, 0, 5, 0, 10, 0, 10, 0, 5, 0, 1])
        mls = "yes"
    elif polyn == 17:                                       # mult
        # nn x alpha
        if npoly is None:
            npoly = 20
        poly = mpmpc([0] * (npoly + 1))
        alpha = TWO - J * TWO
        for kki in np.arange(npoly + 1):
            poly[kki] = mp.binomial(npoly, kki * ONE) * (-alpha) ** kki
        mls = "yes"
    elif polyn >= 20 and polyn <= 80:                       # mult
        npoly = polyn - 20
        poly = mpmpc([0] * (npoly + 1))
        alpha = TWO - J * TWO
        for kki in np.arange(npoly + 1):
            poly[kki] = mp.binomial(npoly, kki * ONE) * (-alpha) ** kki
        mls = "yes"
# http://www-sop.inria.fr/saga/POL/BASE/1.unipol/kameny0.1.7.dir/index.html
    elif polyn == 81:                   # Kameny Polynomial 1
        cnst = TEN ** (-6)
        poly = mpmpc([J * cnst, 0, 0, 0, 0, 1, -six * cnst * cnst,
                      nine * cnst ** 4])
# http://www-sop.inria.fr/saga/POL/BASE/1.unipol/kameny1.1.9.dir/index.html
    elif polyn == 82:                   # Kameny Polynomial 2
        cnst = TEN ** 6
        poly = mpmpc([J * cnst * cnst, 0, 0, 0, 0, cnst ** 4, 0,
                      -six * cnst * cnst, 0, nine])
# http://www-sop.inria.fr/saga/POL/BASE/1.unipol/kameny2.1.9.dir/index.html
    elif polyn == 83:                   # Kameny Polynomial 3
        cnst = TEN ** 6
        poly = mpmpc([cnst * cnst, 0, 0, 0, 0, cnst ** 4, 0,
                      -six * cnst * cnst, 0, nine])
# http://www-sop.inria.fr/saga/POL/BASE/1.unipol/kameny3.1.14.dir/index.html
    elif polyn == 84:                   # Kameny Polynomial 4
        poly = mpmpc([1, 0, 0, TWO * TEN ** 24, 0, 0, TEN ** 48, four, 0, 0,
                      -four * TEN ** 24, 0, 0, 0, four])
# http://www-sop.inria.fr/saga/POL/BASE/1.unipol/large1.1.n.dir/index.html
    elif polyn == 85:                   # Polynomial having large coefs 1
        if npoly is None:
            npoly = 20
        if npoly < 14:
            npoly = 14
        poly = mpmpc([0] * (npoly + 1))
        apoly = ONE                     # n = 20 a = 1, i
        poly[0] = apoly                 # n
        poly[npoly - 14] = TEN ** 300   # 14
        poly[npoly - 5] = ONE           # 5
        poly[npoly] = ONE               # 0
# http://www-sop.inria.fr/saga/POL/BASE/1.unipol/large2.1.n.dir/index.html
    elif polyn == 86:                   # Polynomial having large coefs 2
        if npoly is None:
            npoly = 20
        if npoly < 11:
            npoly = 11
        poly = mpmpc([0] * (npoly + 1))
        apoly = ONE                     # n = 20 a = 1, i
        poly[0] = apoly                 # n
        poly[npoly - 11] = ONE          # 11
        poly[npoly - 1] = TEN ** 300    # 14
        poly[npoly] = TEN ** (-300)     # 0
# http://www-sop.inria.fr/saga/POL/BASE/1.unipol/large3.1.n.dir/index.html
    elif polyn == 87:                 # Polynomial having large coefs 3
        # numpy.linalg.linalg.LinAlgError: Array must not contain infs or NaNs
        # This means that "roots" can't be called on this polynomial.
        # This does not converge with any good values -- msf = 392
        if npoly is None:
            npoly = 20
        poly = mpmpc([0] * (npoly + 1))
        apoly = ONE                     # n = 20 a = 1, i
        poly[0] = apoly * TEN ** (-200)  # n
        poly[1] = TEN ** 100            # n-1
        poly[npoly] = TEN ** 200        # 0
        fmt = 'P[%d]=%+20.12e %+20.12e j;'
        for idx, pidx in enumerate(poly):
            ec_(fmt % (idx, pidx.real, pidx.imag), False)
# http://www-sop.inria.fr/saga/POL/BASE/1.unipol/lgsmclust.1.n.dir/index.html
    elif polyn == 88:      # Poly with large and small clustered roots.
        p01 = [TEN ** 20, -ONE]
        p01 = np.polymul(p01, p01)
        p01 = np.polymul(p01, p01)

        p02 = mpmpc([0] * 13)
        p02[0] = ONE
        p02 = np.polysub(p02, p01)

        p03 = [ONE, TEN ** 20]
        p03 = np.polymul(p03, p03)
        p03 = np.polymul(p03, p03)

        p04 = mpmpc([0] * 9)
        p04[0] = ONE
        p04 = np.polymul(p03, p04)
        p04 = np.polyadd(p04, [ONE])
        poly = np.polymul(p02, p04)
# http://www-sop.inria.fr/saga/POL/BASE/1.unipol/mignotte1.1.n.dir/index.html
    elif polyn == 89:                   # Mignotte-like Polynomial 1
        # Interesting situations are:  large degree and small cluster,
        # large deg and small cluster, i.e., n = 100,500, m =  3, a = 100i
        # large deg and large cluster, i.e., n = 100,500, m = 31, a = 1000
        npoly, mpoly, apoly = [100, 3, J * 100]
        npoly, mpoly, apoly = [100, 31, 1000]
        npoly, mpoly, apoly = [20, 3, J * 100]

        p01 = [apoly, ONE]
        p02 = [ONE]
        for jji in range(mpoly):
            p02 = np.polymul(p02, p01)
        p01 = mpmpc([0] * (npoly + 1))
        p01[0] = ONE
        poly = np.polyadd(p01, p02)
# http://www-sop.inria.fr/saga/POL/BASE/1.unipol/mignotte2.1.n.dir/index.html
    elif polyn == 90:                   # Mignotte-like Polynomial 2
        # n >> m > 1, n >> k > 1, |a| > 1
        npoly, mpoly, kpoly, apoly = [100, 10, 2, 10000]
        npoly, mpoly, kpoly, apoly = [500, 10, 2, 10000]
        npoly, mpoly, kpoly, apoly = [20, 3, 3, J * 100]

        p01 = [apoly, ONE]
        p02 = [ONE]
        for jji in range(mpoly):
            p02 = np.polymul(p02, p01)
        p03 = [apoly, ONE]
        p04 = [ONE]
        for jji in range(kpoly):
            p04 = np.polymul(p04, p03)
        poly = mpmpc([0] * (npoly - kpoly + 1))
        poly[0] = ONE
        poly = np.polyadd(poly, p02)
        poly = np.polymul(poly, p04)
# http://www-sop.inria.fr/saga/POL/BASE/1.unipol/spiral.1.n.dir/index.html
    elif polyn == 91:       # Roots in geometric progression along a spiral
        # |a| << 1 (n = 20, a = J/1000, a = (3 + J * 4) / 10)
        # a = J / 1000  has issues
        if npoly is None:
            npoly = 20
        apoly = (3 + J * 4) / 10
        poly = [ONE]
        fmt = 'znew[%2d] = %+20.12e %+20.12e j;'
        for jji in range(npoly):
            znew = - (apoly ** (jji + 1) - ONE) / (apoly - ONE)
            ec_(fmt % (jji, znew.real, znew.imag), False)
            poly = np.polymul(poly, [ONE, (apoly ** (jji + 1) - ONE) /
                                     (apoly - ONE)])
# http://www-sop.inria.fr/saga/POL/BASE/1.unipol/wilkinson1.1.30.dir/index.html
    elif polyn == 92:                   # Modified Wilkinson's Poly 1
        if npoly is None:
            npoly = 20
        p01 = [ONE]
        for jji in range(npoly):
            p01 = np.polymul(p01, [ONE, mp.mpf(jji + 1)])
        poly = mpmpc([1e-20, 0, 0, 0, 0, 0, 0, 0, 1, -20, 100])
        poly = np.polymul(poly, p01)
# http://www-sop.inria.fr/saga/POL/BASE/1.unipol/wilkinson2.1.22.dir/index.html
    elif polyn == 93:                   # Modified Wilkinson's Poly 2
        if npoly is None:
            npoly = 20                  # (x-20) ** 2
        poly = [ONE]
        for jji in range(npoly):
            poly = np.polymul(poly, [ONE, -mp.mpf(jji + 1)])
        poly = np.polymul(poly, [ONE, -mp.mpf(40), mp.mpf(400)])
        mls = "yes"
# http://www-sop.inria.fr/saga/POL/BASE/1.unipol/expo.1.n.dir/index.html
    elif polyn >= 100 and polyn < 200:  # truncated exponential
        k = 200
        npoly = polyn - (k - 100)
        if polyn == k - 4:
            npoly = 100
        if polyn == k - 3:
            npoly = 500                 # this one doesn't seem to work
        if polyn == k - 2:
            npoly = 1000
        if polyn == k - 1:
            npoly = 2000
        poly = mpmpc([0] * (npoly + 1))
        for kki in range(npoly + 1):
            poly[npoly - kki] = ONE / mp.factorial(kki)
    elif polyn == 203:                  # Chebyshev T3
        poly = mpmpc([4, 0, -3])
    elif polyn == 204:                  # Chebyshev T4
        poly = mpmpc([8, 0, -8, 0, 1])
    elif polyn == 205:                  # Chebyshev T5
        poly = mpmpc([16, 0, -20, 0, 5])
    elif polyn == 206:                  # Chebyshev T6
        poly = mpmpc([32, 0, -48, 0, 18, 0, -1])
    elif polyn == 207:                  # Chebyshev T7
        poly = mpmpc([64, 0, -112, 0, 56, 0, -7])
    elif polyn == 208:                  # Chebyshev T8
        poly = mpmpc([128, 0, -256, 0, 160, 0, -32, 0, 1])
    elif polyn == 209:                  # Chebyshev T9
        poly = mpmpc([256, 0, -576, 0, 432, 0, -120, 0, 9])
    elif polyn == 210:                  # Chebyshev T10
        poly = mpmpc([512, 0, -1280, 0, 1120, 0, -400, 0, 50, 0, -1])
    elif polyn == 211:                  # Chebyshev T11
        poly = mpmpc([1024, 0, -2816, 0, 2816, 0, -1232, 0, 220, 0, -11])
    elif polyn == 303:                  # Chebyshev U3
        poly = mpmpc([8, 0, -4])
    elif polyn == 304:                  # Chebyshev U4
        poly = mpmpc([16, 0, -12, 0, 1])
    elif polyn == 305:                  # Chebyshev U5
        poly = mpmpc([32, 0, -32, 0, 6])
    elif polyn == 306:                  # Chebyshev U6
        poly = mpmpc([64, 0, -80, 0, 24, 0, -1])
    elif polyn == 307:                  # Chebyshev U7
        poly = mpmpc([128, 0, -192, 0, 80, 0, -8])
    elif polyn == 308:                  # Chebyshev U8
        poly = mpmpc([256, 0, -448, 0, 240, 0, -40, 0, 1])
    elif polyn == 309:                  # Chebyshev U9
        poly = mpmpc([512, 0, -1024, 0, 672, 0, -160, 0, 10])
    elif polyn == 400:                  # random roots
        nni = ma.floor(np.random.rand() * 60) + 3
        npoly = nni - 1
        poly = np.array(rnd(nni) * 10 ** rnd(nni) +
                        J * rnd(nni) * 10 ** rnd(nni), dtype=mp.mpc)
    elif polyn == 401:                  # random roots with random mult
        mpoly = ma.floor(np.random.rand() * 20) + 3  # 3 to 22 distinct roots
        znew = np.array(rnd(mpoly) * 10 ** rnd(mpoly) +
                        J * rnd(mpoly) * 10 ** rnd(mpoly), dtype=mp.mpc)
        mlt = np.floor(10 ** np.random.rand(mpoly))
        npoly = int(sum(mlt))
        zav = mpmpc([0] * npoly)
        iav = 0
        for idx in range(mpoly):
            for _ in range(int(mlt[idx])):
                zav[iav] = znew[idx]
                iav += 1
        poly = np.poly(zav)
        fmt = "mult:  n=%3d; m=%3d; "
        prnt(fmt % (npoly, mpoly), True)
        print(mlt, flush=True)
        mls = "yes"
    elif polyn == 402:                  # clustered roots
        mpoly = ma.floor(np.random.rand() * 4) + 1  # 1 to 4 clusters
        znew = np.array(rnd(mpoly) * 10 ** rnd(mpoly) +
                        J * rnd(mpoly) * 10 ** rnd(mpoly), dtype=mp.mpc)
        mlt = np.floor(10 ** np.random.rand(mpoly))
        npoly = int(sum(mlt))
        zav = mpmpc([0] * npoly)
        iav = 0
        for idx in range(mpoly):
            zoabs = mp.fabs(znew[idx])
            for _ in range(int(mlt[idx])):
                zav[iav] = znew[idx] + ((np.random.rand() - 0.5) +
                                        J * (np.random.rand() - 0.5)) * \
                    zoabs * mp.mpf(1e-100)
                iav += 1
        poly = np.poly(zav)
        fmt = "clust:  n=%3d; m=%3d; "
        prnt(fmt % (npoly, mpoly), True)
        print(mlt, flush=True)
    else:
        print("unknown polyn=%d" % polyn, flush=True)
        return ZERO, 0, mls

    npoly = np.size(poly) - 1
    return poly, npoly, mls


def rnd(nnn):
    """Generate a random vector:  uniform (-5, +5)."""
    return 10 * (np.random.rand(nnn) - 0.5)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(__doc__)
    else:
        prm_test()
