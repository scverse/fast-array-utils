|pypi| |docs| |ci| |cov| |bench|

.. |pypi| image:: https://img.shields.io/pypi/v/fast-array-utils
   :target: https://pypi.org/project/fast-array-utils/

.. |docs| image:: https://app.readthedocs.com/projects/icb-fast-array-utils/badge/
   :target: https://icb-fast-array-utils.readthedocs-hosted.com/

.. |ci| image:: https://github.com/scverse/fast-array-utils/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/scverse/fast-array-utils/actions/workflows/ci.yml

.. |cov| image:: https://codecov.io/gh/scverse/fast-array-utils/graph/badge.svg?token=CR62H2QRWY
   :target: https://codecov.io/gh/scverse/fast-array-utils

.. |bench| image:: https://img.shields.io/endpoint?url=https://codspeed.io/badge.json
   :target: https://codspeed.io/scverse/fast-array-utils

fast-array-utils
================

.. begin

usage
-----

``fast-array-utils`` supports the following array types:

- ``numpy.ndarray``
- ``scipy.sparse.cs{rc}_{array,matrix}``
- ``cupy.ndarray`` and ``cupyx.scipy.sparse.cs{rc}_matrix``
- ``dask.array.Array``
- ``h5py.Dataset`` and ``zarr.Array``
- ``anndata.abc.CS{CR}Dataset`` (only supported by ``.conv.to_dense`` at the moment)

Use ``fast_array_utils.conv.to_dense`` to densify arrays and optionally move them to CPU memory:

.. code:: python

   from fast_array_utils.conv import to_dense

   numpy_arr = to_dense(sparse_arr_or_mat)
   numpy_arr = to_dense(dask_or_cuda_arr, to_cpu_memory=True)
   dense_dask_arr = to_dense(dask_arr)
   dense_cupy_arr = to_dense(sparse_cupy_mat)

Use ``fast_array_utils.conv.*`` to calculate statistics across one or both axes of a 2D array.
All of them support an `axis` and `dtype` parameter:

.. code:: python

   from fast_array_utils import stats

   all_equal = stats.is_constant(arr_2d)
   col_sums = stats.sum(arr_2d, axis=0)
   mean = stats.mean(arr_2d)
   row_means, row_vars = stats.mean_var(arr_2d, axis=1)

installation
------------

To use ``fast_array_utils.stats`` or ``fast_array_utils.conv``:

.. code:: bash

   (uv) pip install 'fast-array-utils[accel]'

To use ``testing.fast_array_utils``:

.. code:: bash

   (uv) pip install 'fast-array-utils[testing]'
