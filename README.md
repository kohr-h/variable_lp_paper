# variable_lp_paper

Code for the paper [Total variation regularization with variable Lebesgue prior](https://arxiv.org/abs/1702.08807).


## Installation

As of now, a mix of several development branches of [ODL](https://www.github.com/odlgroup/odl) is necessary to run all examples and all implementations. Make sure that you DO NOT install ODL via `pip` or `conda`, those versions will not work.

TODO: provide a simple source (a branch or zipped archive)

### From git source
This library can be installed by cloning the repository and installing with `pip`:

    git clone https://github.com/kohr-h/variable_lp_paper.git
    cd variable_lp_paper
    pip install [--editable] .

With the `--editable` option you can install the library in "developer mode", where local changes take immediate effect (that is, after opening a new console) without re-install.

### From a zipped archive
A quicker option that works without Git is to install from a zipped archive provided by GitHub:

    pip install https://github.com/kohr-h/variable_lp_paper/archive/master.zip


## Optional dependencies

The following packages are not hard dependencies, but they are needed for certain functionality in the library or in the examples:

    conda install cython numba pygpu scipy pillow imageio matplotlib

    # Re-install `variable_lp_paper` to build the Cython extension

    conda install -c astra-toolbox astra-toolbox  # or `scikit-image`

- `cython` provides a Cython implementation of the functionals. After installing it, you also need to re-install this library (in the same way as during the initial installation), in order to build the Cython extensions.
- `numba` activates the Numba implementation(s).
- `pygpu` lets you run native GPU kernels for all functional methods.
- The packages `scipy`, `pillow` and `imageio` are needed in several examples to load, create or manipulate images.
- To run the tomography examples, you need a backend for ray transforms, either `scikit-image` (very slow) or `astra-toolbox` (fast, see [here](https://odlgroup.github.io/odl/getting_started/installing_extensions.html#astra-for-x-ray-tomography) for instructions).
- Graphical output is provided by `matplotlib`.


## Running the examples

The examples are self-contained and need no data other than bundled with the code. Make sure that your working directory is the `examples/` directory since file paths are relative to that location.
