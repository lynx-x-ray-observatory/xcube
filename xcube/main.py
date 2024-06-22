import numpy as np
from astropy import wcs
from astropy.io import fits
from numba import njit, prange

from xcube.utils import mylog


def _read_rmf(rmf):
    with fits.open(rmf, memmap=True) as f:
        if "MATRIX" in f:
            mat_key = "MATRIX"
        elif "SPECRESP MATRIX" in f:
            mat_key = "SPECRESP MATRIX"
        else:
            raise RuntimeError(
                f"Cannot find the response matrix in the RMF "
                f"file {rmf}! It should be named "
                f'"MATRIX" or "SPECRESP MATRIX".'
            )
        mhdu = f[mat_key]
        ehdu = f["EBOUNDS"]
        n_ch = mhdu.header["DETCHANS"]
        num = 0
        ncols = len(mhdu.columns)
        for i in range(1, ncols + 1):
            if mhdu.header[f"TTYPE{i}"] == "F_CHAN":
                num = i
                break
        cmin = mhdu.header.get(f"TLMIN{num}", 1)
        e_min = ehdu.data["E_MIN"]
        e_max = ehdu.data["E_MAX"]
        if "CHANTYPE" in mhdu.header:
            ctype = mhdu.header["CHANTYPE"]
        elif "CHANTYPE" in ehdu.header:
            ctype = ehdu.header["CHANTYPE"]
        else:
            raise KeyError("'CHANTYPE' not specified in RMF!!")
    return e_min, e_max, n_ch, cmin, ctype


@njit(parallel=True)
def _make_cube(data, x, y, cidxs, dx, dy, xmin, ymin, nevent):
    for i in prange(nevent):
        ix = int((x[i] - xmin) / dx)
        iy = int((y[i] - ymin) / dy)
        data[cidxs[i], iy, ix] += 1


class XRaySpectralCube:
    r"""
    Create a spectral cube from an event file and RMF.

    Parameters
    ----------
    evtfile : string or `~astropy.io.fits.HDUList`
        The event file to use.
    rmffile : string or `~astropy.io.fits.HDUList`
        The RMF file to use.
    emin : float, optional
        The minimum energy to include in the cube, in keV.
        Default: None, which is equivalent to 0.0 keV.
    emax : float, optional
        The maximum energy to include in the cube, in keV.
        Default: None, which is equivalent to 100.0 keV.
    tmin : float, optional
        The minimum time to include in the cube, in seconds.
        Default: None, which is equivalent to the minimum time
        in the event file.
    tmax : float, optional
        The maximum time to include in the cube, in seconds.
        Default: None, which is equivalent to the maximum time
        in the event file.
    reblock : integer, optional
        The reblock factor to use. Default: 1.
    width : float, optional
        The width of the cube in terms of the width of the
        region containing the events. Default: None, which
        is equivalent to the full extent of the event file.
    """

    def __init__(
        self,
        evtfile,
        rmffile,
        emin=None,
        emax=None,
        tmin=None,
        tmax=None,
        reblock=1,
        width=None,
    ):
        if not isinstance(reblock, int) or reblock == 0:
            raise ValueError('"reblock" must be an integer and >= 1!')

        # Read the RMF information
        e_min, e_max, n_ch, cmin, ctype = _read_rmf(rmffile)

        # Read the event file
        if isinstance(evtfile, fits.HDUList):
            hdu = evtfile["EVENTS"]
        else:
            hdu = fits.open(evtfile)["EVENTS"]

        # Filter on time if required
        t = hdu.data["TIME"]
        if tmin is None:
            tmin = t.min()
        if tmax is None:
            tmax = t.max()
        idxs = np.logical_and(t >= tmin, t <= tmax)

        # Filter on energy if required
        if emin is None:
            emin = 0.0
        if emax is None:
            emax = 100.0
        eidxs = (e_min > emin) & (e_max < emax)
        ne_bins = eidxs.sum()

        # Figure out which columns have the
        # coordinate information
        xi = None
        yi = None
        ncols = len(hdu.columns)
        for i in range(1, ncols + 1):
            if hdu.header[f"TTYPE{i}"] == "X":
                xi = i
            if hdu.header[f"TTYPE{i}"] == "Y":
                yi = i
            if xi and yi:
                break

        # Get the WCS for the event file.
        w = wcs.WCS(naxis=2)
        w.wcs.ctype = [hdu.header[f"TCTYP{xi}"], hdu.header[f"TCTYP{yi}"]]
        w.wcs.crpix = [hdu.header[f"TCRPX{xi}"], hdu.header[f"TCRPX{yi}"]]
        w.wcs.cunit = [hdu.header[f"TCUNI{xi}"], hdu.header[f"TCUNI{yi}"]]
        w.wcs.crval = [hdu.header[f"TCRVL{xi}"], hdu.header[f"TCRVL{yi}"]]
        w.wcs.cdelt = [hdu.header[f"TCDLT{xi}"], hdu.header[f"TCDLT{yi}"]]

        # Get the bounds of the coordinates
        xmin = hdu.header[f"TLMIN{xi}"]
        ymin = hdu.header[f"TLMIN{yi}"]
        xmax = hdu.header[f"TLMAX{xi}"]
        ymax = hdu.header[f"TLMAX{yi}"]

        xctr = 0.5 * (xmax + xmin)
        yctr = 0.5 * (ymax + ymin)

        sky_center = w.wcs_pix2world(xctr, yctr, hdu.header[f"TLMIN{xi}"])

        offset = 1 if xmin == 1 else 0

        if width is None:
            nx = int(xmax - xmin + offset)
            ny = int(ymax - ymin + offset)
            if offset:
                xmin -= 0.5
                xmax += 0.5
                ymin -= 0.5
                ymax += 0.5
        else:
            xhw = 0.5 * width * (xmax - xmin + offset)
            yhw = 0.5 * width * (ymax - ymin + offset)
            xmin = int(xctr - 0.5 * xhw) - 0.5
            xmax = int(xctr + 0.5 * xhw) + 0.5
            ymin = int(yctr - 0.5 * yhw) - 0.5
            ymax = int(yctr + 0.5 * yhw) + 0.5
            nx = int(xmax - xmin)
            ny = int(ymax - ymin)

        # Determine the map pixel size and number of
        # pixels on a side
        xdel = hdu.header[f"TCDLT{xi}"] * reblock
        ydel = hdu.header[f"TCDLT{yi}"] * reblock
        nx //= reblock
        ny //= reblock

        de = (e_max - e_min)[eidxs]
        emid = 0.5 * (e_min + e_max)[eidxs]
        cbins = np.arange(n_ch) + cmin - 0.5

        c = hdu.data[ctype.upper()][idxs]

        cidxs = np.searchsorted(cbins[eidxs], c)

        good = (cidxs > 0) & (cidxs < ne_bins)

        x = hdu.data["X"][idxs][good].astype("float64")
        y = hdu.data["Y"][idxs][good].astype("float64")

        cidxs = cidxs[good] - 1

        pidxs = np.logical_and(x >= xmin, x <= xmax)
        pidxs &= np.logical_and(y >= ymin, y <= ymax)

        x = x[pidxs]
        y = y[pidxs]
        cidxs = cidxs[pidxs]

        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny

        mylog.info("Making cube.")

        data = np.zeros((ne_bins, ny, nx))

        _make_cube(data, x, y, cidxs, dx, dy, xmin, ymin, x.size)

        mylog.info("Done making cube.")

        cubehdu = fits.PrimaryHDU(data)

        cubehdu.header["CTYPE1"] = w.wcs.ctype[0]
        cubehdu.header["CTYPE2"] = w.wcs.ctype[1]
        cubehdu.header["CTYPE3"] = "ENER"
        cubehdu.header["CRVAL1"] = float(sky_center[0])
        cubehdu.header["CRVAL2"] = float(sky_center[1])
        cubehdu.header["CRVAL3"] = emid[0]
        cubehdu.header["CUNIT1"] = "deg"
        cubehdu.header["CUNIT2"] = "deg"
        cubehdu.header["CUNIT3"] = "keV"
        cubehdu.header["CDELT1"] = xdel
        cubehdu.header["CDELT2"] = ydel
        cubehdu.header["CDELT3"] = de[0]
        cubehdu.header["CRPIX1"] = 0.5 * (nx + 1)
        cubehdu.header["CRPIX2"] = 0.5 * (ny + 1)
        cubehdu.header["CRPIX3"] = 1

        cubehdu.name = "FLUX"

        self.cubehdu = cubehdu

    @property
    def shape(self):
        return self.cubehdu.shape

    def collapse(self):
        data = self.cubehdu.data.sum(axis=0)
        imhdu = fits.PrimaryHDU(data)

        for key in ["CTYPE", "CRVAL", "CUNIT", "CDELT", "CRPIX"]:
            for i in range(1, 3):
                imhdu.header[f"{key}{i}"] = self.cubehdu.header[f"{key}{i}"]

        imhdu.name = "FLUX"

        return imhdu

    def writeto(
        self,
        fileobj,
        output_verify="exception",
        overwrite=False,
        checksum=False,
    ):
        r"""

        Write the cube to a FITS file. The following parameters are the
        same as those for `~astropy.io.fits.HDUList.writeto`.

        Parameters
        ----------
        fileobj : string, file-like or `pathlib.Path`
            File to write to.  If a file object, must be opened in a
            writeable mode.

        output_verify : string
            Output verification option.  Must be one of ``"fix"``,
            ``"silentfix"``, ``"ignore"``, ``"warn"``, or
            ``"exception"``.  May also be any combination of ``"fix"`` or
            ``"silentfix"`` with ``"+ignore"``, ``"+warn"``, or ``"+exception"``
            (e.g. ``"fix+warn"``).

        overwrite : bool, optional
            If ``True``, overwrite the output file if it exists. Raises an
            ``OSError`` if ``False`` and the output file exists. Default is
            ``False``.

        checksum : bool
            When `True` adds both ``DATASUM`` and ``CHECKSUM`` cards
            to the header of the HDU when written to the file.

        Notes
        -----
        gzip, zip and bzip2 compression algorithms are natively supported.
        Compression mode is determined from the filename extension
        ('.gz', '.zip' or '.bz2' respectively).  It is also possible to pass a
        compressed file object, e.g. `gzip.GzipFile`.
        """
        self.cubehdu.writeto(
            fileobj, output_verify=output_verify, overwrite=overwrite, checksum=checksum
        )
