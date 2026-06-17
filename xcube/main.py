import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
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
        data[ix, iy, cidxs[i]] += 1


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

    def __init__(self, filename, hdu_index=0):
        if isinstance(filename, (fits.PrimaryHDU, fits.ImageHDU)):
            data = np.array(filename.data, dtype=float)
            fits_header = filename.header.copy()
        elif isinstance(filename, str):
            with fits.open(filename) as hdul:
                hdu = hdul[hdu_index]
                # hdu.data is lazily loaded, so it must be read inside the
                # context manager — accessing it after the file closes fails.
                data = np.array(hdu.data, dtype=float)
                fits_header = hdu.header.copy()
        else:
            raise TypeError(
                "filename must be a path string or a FITS PrimaryHDU/ImageHDU"
            )

        header = dict(fits_header)  # plain dict — units read before WCS normalises them

        if data.ndim < 3:
            raise ValueError(
                f"HDU {hdu_index} has {data.ndim} dimensions; need at least 3"
            )

        try:
            wcs = WCS(fits_header, naxis=3)
        except Exception:
            wcs = None

        spec_numpy_axis = self._find_spectral_axis(wcs, data.ndim)
        if spec_numpy_axis != data.ndim - 1:
            data = np.moveaxis(data, spec_numpy_axis, -1)

        self._data = data
        self._header = fits_header
        self._wcs = wcs
        # numpy axis the spectral data occupied on input (before the move to
        # last); writeto uses it to restore the axis order the header describes.
        self._spec_numpy_axis = spec_numpy_axis
        self._spectral_fits_axis = (
            int(wcs.wcs.spec) if wcs is not None and wcs.wcs.spec >= 0 else None
        )
        self._spectral_coords = self._build_spectral_coords(header)
        self._spectral_unit = self._read_spectral_unit(header)
        self._channel_width = self._read_channel_width(header)
        self._bunit = str(header.get("BUNIT", "")).strip()

    def _find_spectral_axis(self, wcs, ndim):
        """Return the numpy axis index (0-based) that carries the spectral axis."""
        if wcs is not None and wcs.wcs.spec >= 0:
            # wcs.wcs.spec is 0-based in FITS pixel order (0 == NAXIS1 == fastest).
            # numpy reverses axis order, so numpy_axis = ndim - 1 - fits_axis.
            return ndim - 1 - int(wcs.wcs.spec)
        return ndim - 1  # spectral axis last by convention (e.g. (y, x, spectral))

    def _build_spectral_coords(self, header):
        n = self._data.shape[-1]
        if self._spectral_fits_axis is None:
            return np.arange(n, dtype=float)

        # FITS keywords are 1-based; spectral_fits_axis is 0-based.
        ax = self._spectral_fits_axis + 1
        crpix = float(header.get(f"CRPIX{ax}", 1)) - 1  # convert to 0-based
        crval = float(header.get(f"CRVAL{ax}", 0))
        # Prefer CDELTn; fall back to CD matrix diagonal
        if f"CDELT{ax}" in header:
            cdelt = float(header[f"CDELT{ax}"])
        elif f"CD{ax}_{ax}" in header:
            cdelt = float(header[f"CD{ax}_{ax}"])
        else:
            cdelt = 1.0
        return crval + (np.arange(n, dtype=float) - crpix) * cdelt

    def _read_spectral_unit(self, header):
        if self._spectral_fits_axis is None:
            return "pixel"
        ax = self._spectral_fits_axis + 1
        unit = str(header.get(f"CUNIT{ax}", "")).strip()
        return unit if unit else "pixel"

    def _read_channel_width(self, header):
        """Absolute channel width in spectral_unit units."""
        if self._spectral_fits_axis is None:
            return 1.0
        ax = self._spectral_fits_axis + 1
        if f"CDELT{ax}" in header:
            return abs(float(header[f"CDELT{ax}"]))
        if f"CD{ax}_{ax}" in header:
            return abs(float(header[f"CD{ax}_{ax}"]))
        if self._data.shape[-1] > 1:
            return abs(float(self._spectral_coords[1] - self._spectral_coords[0]))
        return 1.0

    @classmethod
    def from_event_file(
        cls,
        evtfile,
        rmffile,
        emin=None,
        emax=None,
        tmin=None,
        tmax=None,
        reblock=1,
        width=None,
        adaptive_bins=None,
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

        # Filter on energy
        if emin is None:
            emin = 0.0
        if emax is None:
            emax = 100.0

        c = hdu.data[ctype.upper()][idxs]

        channels = np.arange(n_ch) + cmin

        if adaptive_bins is None:
            eidxs = (e_min > emin) & (e_max < emax)
            ne_bins = eidxs.sum()
            e_min = e_min[eidxs]
            e_max = e_max[eidxs]
            cbins = channels[eidxs] - 0.5
        else:
            sigma, max_size = adaptive_bins
            sigma2 = sigma * sigma
            y = np.bincount(c.astype("int"), minlength=n_ch + cmin)[cmin:]
            ebins = []
            cbins = []
            sum = 0.0
            max_size = 0
            for i in range(y.size):
                if e_min[i] > emin and e_max[i] < emax:
                    if len(ebins) == 0:
                        ebins.append(e_min[i])
                        cbins.append(channels[i])
                    else:
                        sum += y[i]
                        max_size += 1
                        if sum >= sigma2 or i == y.size - 1:
                            ebins.append(e_max[i])
                            cbins.append(channels[i] + 1)
                            sum = 0.0
                            max_size = 0
            e_min = np.array(ebins[:-1])
            e_max = np.array(ebins[1:])
            cbins = np.array(cbins) - 0.5
            ne_bins = e_min.size

        de = e_max - e_min
        emid = 0.5 * (e_min + e_max)

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

        # Get the WCS from the event file.
        w = WCS(naxis=2)
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

        cidxs = np.searchsorted(cbins, c)

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

        data = np.zeros((nx, ny, ne_bins))

        _make_cube(data, x, y, cidxs, dx, dy, xmin, ymin, x.size)

        mylog.info("Done making cube.")

        # _make_cube builds (nx, ny, ne) — numpy axes (x, y, energy).  Transpose
        # to (ne, ny, nx) so the FITS axes are NAXIS1=x (RA), NAXIS2=y (Dec),
        # NAXIS3=energy (CTYPE3=ENER): the conventional spectral-cube layout with
        # the celestial WCS axes matching the spatial data (no transpose).
        # __init__ moves the spectral axis back to numpy-last on read, giving
        # internal (ny, nx, ne) so get_slice returns [row=y/Dec, col=x/RA].
        data = np.ascontiguousarray(data.T)

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

        return cls(cubehdu)

    @property
    def shape(self):
        """(ny, nx, n_channels) — spectral axis last."""
        return self._data.shape

    @property
    def n_channels(self):
        return self._data.shape[-1]

    @property
    def spectral_coords(self):
        """1D array of spectral coordinate values (length == n_channels)."""
        return self._spectral_coords

    @property
    def spectral_unit(self):
        """String label for the spectral axis unit."""
        return self._spectral_unit

    @property
    def channel_width(self):
        """Absolute width of one spectral channel in spectral_unit units."""
        return self._channel_width

    @property
    def bunit(self):
        """Raw BUNIT string from the FITS header (spectral density unit)."""
        return self._bunit

    @property
    def spectrum_unit(self):
        """
        Unit of integrated spectra (BUNIT × spectral_unit).

        Derived by stripping the trailing '/<spectral_unit>' from BUNIT if
        present; e.g. 'count/s/keV' with spectral_unit='keV' → 'count/s'.
        """
        suffix = f"/{self._spectral_unit}"
        if self._bunit.endswith(suffix):
            return self._bunit[: -len(suffix)]
        return self._bunit or "intensity"

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
        # The spectral axis was moved to numpy-last on read; restore the original
        # axis order so the data matches the axis description in self._header.
        out_data = np.ascontiguousarray(
            np.moveaxis(self._data, -1, self._spec_numpy_axis)
        )
        cubehdu = fits.PrimaryHDU(data=out_data, header=self._header)
        cubehdu.writeto(
            fileobj, output_verify=output_verify, overwrite=overwrite, checksum=checksum
        )

    def get_slice(self, channel):
        """Return the 2D (ny, nx) image at spectral channel *channel*."""
        return self._data[..., channel]

    def get_spectrum(self, xi, yi):
        """Return the 1D spectrum (counts/s) at spatial pixel (xi, yi)."""
        return self._data[int(yi), int(xi), :]

    def get_integrated_spectrum(self):
        """Return the spectrum (counts/s) summed over all spatial pixels."""
        return np.nansum(self._data, axis=(0, 1))

    def get_region_spectrum(self, mask):
        """Return the spectrum (counts/s) summed over pixels where *mask* is True."""
        return np.nansum(self._data[mask], axis=0)

    def get_collapsed_image(self, e_min, e_max):
        """
        Return a 2D (ny, nx) image (counts/s) summed over spectral channels
        within [e_min, e_max].  Returns None if no channels fall in range.
        """
        idx = np.where(
            (self._spectral_coords >= e_min) & (self._spectral_coords <= e_max)
        )[0]
        if idx.size == 0:
            return None
        return np.nansum(self._data[..., idx], axis=-1)
