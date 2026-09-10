"""
Routines to find critical points (O- and X-points)

Copyright 2016 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

This file is part of FreeGS.

FreeGS is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGS.  If not, see <http://www.gnu.org/licenses/>.

"""


from scipy import interpolate, ndimage
from numpy import zeros
from numpy.linalg import inv
from numpy import (
    dot,
    linspace,
    argmax,
    argmin,
    abs,
    clip,
    sin,
    cos,
    pi,
    amax,
    arctan2,
    sqrt,
    sum,
)
import numpy as np
from warnings import warn


def find_critical(R, Z, psi, discard_xpoints=True):
    """
    Find critical points

    Inputs
    ------
    R:
        R(nr, nz) 2D array of major radii
    Z:
        Z(nr, nz) 2D array of heights
    psi:
        psi(nr, nz) 2D array of psi values

    Returns
    -------
    opoint: list
        List of O-points (magnetic axes) consisting of ``(R, Z, psi)`` tuples
    xpoint: list
        List of X-points (magnetic axes) consisting of ``(R, Z, psi)`` tuples

    """

    # Get a spline interpolation function
    f = interpolate.RectBivariateSpline(R[:, 0], Z[0, :], psi)

    # Find candidate locations, based on minimising Bp^2
    #
    # R and Z come from meshgrid, so the spline can be evaluated as a tensor
    # product on the 1D axes rather than at every (R, Z) pair separately. It is
    # the same tensor product either way and returns bit-identical values, but
    # the scattered form costs ~22x more (0.130 s against 0.006 s at 257x513).
    Bp2 = (
        f(R[:, 0], Z[0, :], dx=1, grid=True) ** 2
        + f(R[:, 0], Z[0, :], dy=1, grid=True) ** 2
    ) / R ** 2

    # Get grid resolution, which determines a reasonable tolerance
    # for the Newton iteration search area
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]
    radius_sq = 9 * (dR ** 2 + dZ ** 2)

    # Find local minima
    #
    # The eight-neighbour test as eight array comparisons on shifted slices,
    # rather than a Python loop over every interior point -- the same test, but
    # it no longer costs O(nx*ny) interpreted iterations. The window is still
    # [2, n-2) because the O/X classification below uses a second-neighbour
    # stencil.

    J = zeros([2, 2])

    xpoint = []
    opoint = []

    nx, ny = Bp2.shape
    interior = Bp2[2:-2, 2:-2]
    is_min = np.ones(interior.shape, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            is_min &= interior < Bp2[2 + di:nx - 2 + di, 2 + dj:ny - 2 + dj]

    # Row-major, so candidates are visited in the same order as the old loop.
    for i, j in zip(*(idx + 2 for idx in np.nonzero(is_min))):

        # Found local minimum

        R0 = R[i, j]
        Z0 = Z[i, j]

        # Use Newton iterations to find where
        # both Br and Bz vanish
        R1 = R0
        Z1 = Z0

        count = 0
        while True:

            Br = -f(R1, Z1, dy=1, grid=False) / R1
            Bz = f(R1, Z1, dx=1, grid=False) / R1

            if Br ** 2 + Bz ** 2 < 1e-6:
                # Found a minimum. Classify as either
                # O-point or X-point

                dR = R[1, 0] - R[0, 0]
                dZ = Z[0, 1] - Z[0, 0]
                d2dr2 = (psi[i + 2, j] - 2.0 * psi[i, j] + psi[i - 2, j]) / (
                    2.0 * dR
                ) ** 2
                d2dz2 = (psi[i, j + 2] - 2.0 * psi[i, j] + psi[i, j - 2]) / (
                    2.0 * dZ
                ) ** 2
                d2drdz = (
                    (psi[i + 2, j + 2] - psi[i + 2, j - 2]) / (4.0 * dZ)
                    - (psi[i - 2, j + 2] - psi[i - 2, j - 2]) / (4.0 * dZ)
                ) / (4.0 * dR)
                D = d2dr2 * d2dz2 - d2drdz ** 2

                if D < 0.0:
                    # Found X-point
                    xpoint.append((R1, Z1, f(R1, Z1)[0][0]))
                else:
                    # Found O-point
                    opoint.append((R1, Z1, f(R1, Z1)[0][0]))
                break

            # Jacobian matrix
            # J = ( dBr/dR, dBr/dZ )
            #     ( dBz/dR, dBz/dZ )

            J[0, 0] = -Br / R1 - f(R1, Z1, dy=1, dx=1)[0][0] / R1
            J[0, 1] = -f(R1, Z1, dy=2)[0][0] / R1
            J[1, 0] = -Bz / R1 + f(R1, Z1, dx=2) / R1
            J[1, 1] = f(R1, Z1, dx=1, dy=1)[0][0] / R1

            d = dot(inv(J), [Br, Bz])

            R1 = R1 - d[0]
            Z1 = Z1 - d[1]

            count += 1
            # If (R1,Z1) is too far from (R0,Z0) then discard
            # or if we've taken too many iterations
            if ((R1 - R0) ** 2 + (Z1 - Z0) ** 2 > radius_sq) or (count > 100):
                # Discard this point
                break

    # Remove duplicates
    def remove_dup(points):
        result = []
        for n, p in enumerate(points):
            dup = False
            for p2 in result:
                if (p[0] - p2[0]) ** 2 + (p[1] - p2[1]) ** 2 < 1e-5:
                    dup = True  # Duplicate
                    break
            if not dup:
                result.append(p)  # Add to the list
        return result

    xpoint = remove_dup(xpoint)
    opoint = remove_dup(opoint)

    if len(opoint) == 0:
        # Can't order primary O-point, X-point so return
        print("Warning: No O points found")
        return opoint, xpoint

    # Find primary O-point by sorting by distance from middle of domain
    Rmid = 0.5 * (R[-1, 0] + R[0, 0])
    Zmid = 0.5 * (Z[0, -1] + Z[0, 0])
    opoint.sort(key=lambda x: (x[0] - Rmid) ** 2 + (x[1] - Zmid) ** 2)

    # Draw a line from the O-point to each X-point. Psi should be
    # monotonic; discard those which are not

    if discard_xpoints:
        Ro, Zo, Po = opoint[0]  # The primary O-point
        xpt_keep = []
        for xpt in xpoint:
            Rx, Zx, Px = xpt

            rline = linspace(Ro, Rx, num=50)
            zline = linspace(Zo, Zx, num=50)

            pline = f(rline, zline, grid=False)

            if Px < Po:
                pline *= -1.0  # Reverse, so pline is maximum at X-point

            # Now check that pline is monotonic
            # Tried finding maximum (argmax) and testing
            # how far that is from the X-point. This can go
            # wrong because psi can be quite flat near the X-point
            # Instead here look for the difference in psi
            # rather than the distance in space

            maxp = amax(pline)
            if (maxp - pline[-1]) / (maxp - pline[0]) > 0.001:
                # More than 0.1% drop in psi from maximum to X-point
                # -> Discard
                continue

            ind = argmin(pline)  # Should be at O-point
            if (rline[ind] - Ro) ** 2 + (zline[ind] - Zo) ** 2 > 1e-4:
                # Too far, discard
                continue
            xpt_keep.append(xpt)
        xpoint = xpt_keep

    # Sort X-points by distance to primary O-point in psi space
    psi_axis = opoint[0][2]
    xpoint.sort(key=lambda x: (x[2] - psi_axis) ** 2)

    return opoint, xpoint


def core_mask(R, Z, psi, opoint, xpoint=[], psi_bndry=None):
    """
    Mark the parts of the domain which are in the core

    Inputs
    ------

    R[nx,ny]:
        2D array of major radius (R) values
    Z[nx,ny]:
        2D array of height (Z) values
    psi[nx,ny]:
        2D array of poloidal flux
    opoint, xpoint :
        Values returned by find_critical

    If psi_bndry is not None, then that is used to find the
    separatrix, not the X-points.

    Returns
    -------
    numpy.ndarray
        A 2D array [nx,ny] which is 1 inside the core, 0 outside

    """

    mask = zeros(psi.shape)
    nx, ny = psi.shape

    # Start and end points
    Ro, Zo, psi_axis = opoint[0]
    if psi_bndry is None:
        _, _, psi_bndry = xpoint[0]

    # Normalise psi
    psin = (psi - psi_axis) / (psi_bndry - psi_axis)

    # Need some care near X-points to avoid flood filling through saddle point
    # Here we first block off the x-point regions, so the fill cannot leak
    # through the saddle, then later return to handle these more difficult cases
    #
    blocked = np.zeros(psi.shape, dtype=bool)
    xpt_inds = []
    for rx, zx, _ in xpoint:
        # Find nearest index
        ix = argmin(abs(R[:, 0] - rx))
        jx = argmin(abs(Z[0, :] - zx))
        xpt_inds.append((ix, jx))
        # Block this point and all around it
        blocked[np.clip(ix - 1, 0, nx - 1):np.clip(ix + 2, 0, nx),
                np.clip(jx - 1, 0, ny - 1):np.clip(jx + 2, 0, ny)] = True

    # Find nearest index to start
    rind = argmin(abs(R[:, 0] - Ro))
    zind = argmin(abs(Z[0, :] - Zo))

    # The core is the connected region of {psi_n < 1} containing the O-point.
    #
    # This replaces a row-scanning flood fill written in Python, which cost
    # ~0.63 s at 257x513 and is called up to three times per Picard iteration.
    # That fill only ever steps to (i +- 1, j) or (i, j +- 1), so the region it
    # marks is exactly the 4-connected component ndimage.label finds with its
    # default structure -- verified array_equal on a 257x513 STEP equilibrium.
    #
    # One deliberate difference, in a case that cannot arise physically: the
    # old fill marked the O-point cell without first testing psi_n there, so a
    # seed outside {psi_n < 1} would still produce a non-empty mask. Here such
    # a seed lands in no component and the mask comes back empty. psi_n is 0 at
    # the O-point by construction, so this only changes behaviour for input
    # where opoint does not describe the psi array it was passed with.
    labels, _ = ndimage.label(np.logical_and(psin < 1.0, ~blocked))
    seed = labels[rind, zind]
    if seed:
        mask = np.where(labels == seed, 1.0, 0.0)

    # Now return to X-point locations
    for ix, jx in xpt_inds:
        for i in np.clip([ix - 1, ix, ix + 1], 0, nx - 1):
            for j in np.clip([jx - 1, jx, jx + 1], 0, ny - 1):
                if psin[i, j] < 1.0:
                    mask[i, j] = 1
                else:
                    mask[i, j] = 0

    return mask


def find_psisurface(eq, psifunc, r0, z0, r1, z1, psival=1.0, n=100, axis=None):
    """
    eq      - Equilibrium object
    (r0,z0) - Start location inside separatrix
    (r1,z1) - Location outside separatrix

    n - Number of starting points to use
    """
    # Clip (r1,z1) to be inside domain
    # Shorten the line so that the direction is unchanged
    if abs(r1 - r0) > 1e-6:
        rclip = clip(r1, eq.Rmin, eq.Rmax)
        z1 = z0 + (z1 - z0) * abs((rclip - r0) / (r1 - r0))
        r1 = rclip

    if abs(z1 - z0) > 1e-6:
        zclip = clip(z1, eq.Zmin, eq.Zmax)
        r1 = r0 + (r1 - r0) * abs((zclip - z0) / (z1 - z0))
        z1 = zclip

    r = linspace(r0, r1, n)
    z = linspace(z0, z1, n)

    if axis is not None:
        axis.plot(r, z)

    pnorm = psifunc(r, z, grid=False)

    if hasattr(psival, "__len__"):
        pass

    else:
        # Only one value
        ind = argmax(pnorm > psival)

        if ind == 0:
            # If the point is very close to the magnetic axis, don't
            # try to do extrapolation.
            r = r[ind]
            z = z[ind]
        else:
            # Edited by Bhavin 31/07/18
            # Changed 1.0 to psival in f
            # make f gradient to psival surface
            f = (pnorm[ind] - psival) / (pnorm[ind] - pnorm[ind - 1])
            
            # Interpolate between points
            r = (1.0 - f) * r[ind] + f * r[ind - 1]
            z = (1.0 - f) * z[ind] + f * z[ind - 1]

            if f > 1.0: warn(f"find_psisurface has encountered an extrapolation. This will probably result in a point where you don't expect it.")

    if axis is not None:
        axis.plot(r, z, "bo")

    return r, z


def find_separatrix(
    eq, opoint=None, xpoint=None, ntheta=20, psi=None, axis=None, psival=1.0
):
    """Find the R, Z coordinates of the separatrix for equilbrium
    eq. Returns a tuple of (R, Z, R_X, Z_X), where R_X, Z_X are the
    coordinates of the X-point on the separatrix. Points are equally
    spaced in geometric poloidal angle.

    If opoint, xpoint or psi are not given, they are calculated from eq

    eq - Equilibrium object
    opoint - List of O-point tuples of (R, Z, psi)
    xpoint - List of X-point tuples of (R, Z, psi)
    ntheta - Number of points to find
    psi - Grid of psi on (R, Z)
    axis - A matplotlib axis object to plot points on
    """
    if psi is None:
        psi = eq.psi()

    if (opoint is None) or (xpoint is None):
        opoint, xpoint = find_critical(eq.R, eq.Z, psi)

    psinorm = (psi - opoint[0][2]) / (eq.psi_bndry - opoint[0][2])

    psifunc = interpolate.RectBivariateSpline(eq.R[:, 0], eq.Z[0, :], psinorm)

    r0, z0 = opoint[0][0:2]

    theta_grid = linspace(0, 2 * pi, ntheta, endpoint=False)
    dtheta = theta_grid[1] - theta_grid[0]

    # Avoid putting theta grid points exactly on the X-points
    xpoint_theta = arctan2(xpoint[0][0] - r0, xpoint[0][1] - z0)
    xpoint_theta = xpoint_theta * (xpoint_theta >= 0) + (xpoint_theta + 2 * pi) * (
        xpoint_theta < 0
    )  # let's make it between 0 and 2*pi
    # How close in theta to allow theta grid points to the X-point
    TOLERANCE = 1.0e-3
    if any(abs(theta_grid - xpoint_theta) < TOLERANCE):
        warn("Theta grid too close to X-point, shifting by half-step")
        theta_grid += dtheta / 2

    isoflux = []
    for theta in theta_grid:
        r, z = find_psisurface(
            eq,
            psifunc,
            r0,
            z0,
            r0 + 10.0 * sin(theta),
            z0 + 10.0 * cos(theta),
            psival=psival,
            axis=axis,
            n=1000,
        )
        isoflux.append((r, z, xpoint[0][0], xpoint[0][1]))

    return isoflux


def find_safety(
    eq, npsi=1, psinorm=None, ntheta=128, psi=None, opoint=None, xpoint=None, axis=None
):
    """Find the safety factor for each value of psi
    Calculates equally spaced flux surfaces. Points on
    each flux surface are equally paced in poloidal angle
    Performs line integral around flux surface to get q

    eq - The equilbrium object
    psinorm flux surface to calculate it for
    npsi - Number of flux surface values to find q for
    ntheta - Number of poloidal points to find it on

    If opoint, xpoint or psi are not given, they are calculated from eq

    returns safety factor for npsi points in normalised psi
    """

    if psi is None:
        psi = eq.psi()

    if (opoint is None) or (xpoint is None):
        opoint, xpoint = find_critical(eq.R, eq.Z, psi)

    if (xpoint is None) or (len(xpoint) == 0):
        # No X-point
        raise ValueError("No X-point so no separatrix")
    else:
        psinormal = (psi - opoint[0][2]) / (xpoint[0][2] - opoint[0][2])

    psifunc = interpolate.RectBivariateSpline(eq.R[:, 0], eq.Z[0, :], psinormal)

    r0, z0 = opoint[0][0:2]

    theta_grid = linspace(0, 2 * pi, ntheta, endpoint=False)
    dtheta = theta_grid[1] - theta_grid[0]

    # Avoid putting theta grid points exactly on the X-points
    xpoint_theta = arctan2(xpoint[0][0] - r0, xpoint[0][1] - z0)
    xpoint_theta = xpoint_theta * (xpoint_theta >= 0) + (xpoint_theta + 2 * pi) * (
        xpoint_theta < 0
    )  # let's make it between 0 and 2*pi
    # How close in theta to allow theta grid points to the X-point
    TOLERANCE = 1.0e-3

    if any(abs(theta_grid - xpoint_theta) < TOLERANCE):
        warn("Theta grid too close to X-point, shifting by half-step")
        theta_grid += dtheta / 2

    if psinorm is None:
        npsi = 100
        psirange = linspace(1.0 / (npsi + 1), 1.0, npsi, endpoint=False)
    else:
        try:
            psirange = psinorm
            npsi = len(psinorm)
        except TypeError:
            npsi = 1
            psirange = [psinorm]

    psisurf = zeros([npsi, ntheta, 2])

    # Calculate flux surface positions
    #
    # The ray searched by find_psisurface depends on theta ALONE -- psival only
    # enters afterwards, when the crossing is located along it. Looping psi
    # outside theta therefore re-sampled the same 100-point ray once per flux
    # surface: at the npsi=100, ntheta=128 defaults that is 12,800 spline
    # evaluations where 128 carry all the information, and it dominated the cost
    # of writing a G-EQDSK (whose qpsi column comes through here). Sampling each
    # ray once and locating every surface's crossing on it together leaves the
    # returned q bit-identical, at ~50x less work.
    n = 100  # find_psisurface's own sampling density along the ray
    psirange_arr = np.asarray(psirange, dtype=float)
    for j in range(ntheta):
        theta = theta_grid[j]
        r1 = r0 + np.ptp(eq.R) * sin(theta)
        z1 = z0 + np.ptp(eq.Z) * cos(theta)

        # Clip the far end into the domain, shortening the line so that the
        # direction is unchanged -- as find_psisurface does.
        if abs(r1 - r0) > 1e-6:
            rclip = clip(r1, eq.Rmin, eq.Rmax)
            z1 = z0 + (z1 - z0) * abs((rclip - r0) / (r1 - r0))
            r1 = rclip
        if abs(z1 - z0) > 1e-6:
            zclip = clip(z1, eq.Zmin, eq.Zmax)
            r1 = r0 + (r1 - r0) * abs((zclip - z0) / (z1 - z0))
            z1 = zclip

        rr = linspace(r0, r1, n)
        zz = linspace(z0, z1, n)
        if axis is not None:
            axis.plot(rr, zz)

        pnorm = psifunc(rr, zz, grid=False)

        # First index where pnorm exceeds each psival. argmax on a boolean
        # returns the first True, so this is the scalar argmax of the original
        # done for every surface at once -- exact even where pnorm is not
        # monotonic along the ray.
        ind = argmax(pnorm[None, :] > psirange_arr[:, None], axis=1)

        # ind == 0 means the surface is very close to the magnetic axis; the
        # original declines to extrapolate there and takes the first point.
        at_axis = ind == 0
        ind_hi = np.clip(ind, 1, n - 1)
        denom = pnorm[ind_hi] - pnorm[ind_hi - 1]
        frac = np.where(denom != 0.0, (pnorm[ind_hi] - psirange_arr) / denom, 0.0)

        psisurf[:, j, 0] = np.where(
            at_axis, rr[0], (1.0 - frac) * rr[ind_hi] + frac * rr[ind_hi - 1]
        )
        psisurf[:, j, 1] = np.where(
            at_axis, zz[0], (1.0 - frac) * zz[ind_hi] + frac * zz[ind_hi - 1]
        )

        if axis is not None:
            axis.plot(psisurf[:, j, 0], psisurf[:, j, 1], "bo")

    # Get variables for loop integral around flux surface
    r = psisurf[:, :, 0]
    z = psisurf[:, :, 1]
    fpol = eq.fpol(psirange[:]).reshape(npsi, 1)
    Br = eq.Br(r, z)
    Bz = eq.Bz(r, z)
    Bthe = sqrt(Br ** 2 + Bz ** 2)

    # Differentiate location w.r.t. index
    dr_di = (np.roll(r, 1, axis=1) - np.roll(r, -1, axis=1)) / 2.0
    dz_di = (np.roll(z, 1, axis=1) - np.roll(z, -1, axis=1)) / 2.0

    # Distance between points
    dl = sqrt(dr_di ** 2 + dz_di ** 2)

    # Integrand - Btor/(R*Bthe) = Fpol/(R**2*Bthe)
    qint = fpol / (r ** 2 * Bthe)

    # Integral
    q = sum(qint * dl, axis=1) / (2 * pi)

    return q
