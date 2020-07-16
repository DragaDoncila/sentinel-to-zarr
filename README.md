# SENTINEL 2 ZARR

![Python Logo](https://www.python.org/static/community_logos/python-logo.png "Sample inline image")

A small utility to convert collections of [ESA Sentinel](https://sentinel.esa.int) tile data into multiscale zarr for fast and easy browsing.
For now we are using the [OME-Zarr metadata spec](https://github.com/ome/omero-ms-zarr/blob/master/spec.md), but in the future we probably want to use a more geoscience-specific spec, see eg [this thread](https://github.com/zarr-developers/zarr-specs/issues/23#issuecomment-490505237) for discussion.