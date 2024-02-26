"""Gaussian descriptor class."""
import os

import ase
import ase.io
from ase.neighborlist import NeighborList
try:
    from lammps import lammps
    # For version compatibility; older lammps versions (the serial version
    # we still use on some machines) do not have these constants.
    try:
        from lammps import constants as lammps_constants
    except ImportError:
        pass
except ModuleNotFoundError:
    pass
import numpy as np

from mala.descriptors.lammps_utils import set_cmdlinevars, extract_compute_np
from mala.descriptors.descriptor import Descriptor

# Empirical value for the Gaussian descriptor width, determined for an
# aluminium system. Reasonable values for sigma can and will be calculated
# automatically based on this value and the aluminium gridspacing
# for other systems as well.
optimal_sigma_aluminium = 0.2
reference_grid_spacing_aluminium = 0.08099000022712448


class AtomicDensity(Descriptor):
    """Class for calculation and parsing of Gaussian descriptors.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters object used to create this object.
    """

    def __init__(self, parameters):
        super(AtomicDensity, self).__init__(parameters)
        self.verbosity = parameters.verbosity

    @property
    def data_name(self):
        """Get a string that describes the target (for e.g. metadata)."""
        return "AtomicDensity"

    @property
    def feature_size(self):
        """Get the feature dimension of this data."""
        return self.fingerprint_length

    @staticmethod
    def convert_units(array, in_units="None"):
        """
        Convert the units of a Gaussian descriptor.

        Since these do not really have units this function does nothing yet.

        Parameters
        ----------
        array : numpy.array
            Data for which the units should be converted.

        in_units : string
            Units of array.

        Returns
        -------
        converted_array : numpy.array
            Data in MALA units.
        """
        if in_units == "None" or in_units is None:
            return array
        else:
            raise Exception("Unsupported unit for Gaussian descriptors.")

    @staticmethod
    def backconvert_units(array, out_units):
        """
        Convert the units of a Gaussian descriptor.

        Since these do not really have units this function does nothing yet.

        Parameters
        ----------
        array : numpy.array
            Data in MALA units.

        out_units : string
            Desired units of output array.

        Returns
        -------
        converted_array : numpy.array
            Data in out_units.
        """
        if out_units == "None":
            return array
        else:
            raise Exception("Unsupported unit for Gaussian descriptors.")

    @staticmethod
    def get_optimal_sigma(voxel):
        """
        Calculate the optimal width of the Gaussians based on the grid voxel.

        Parameters
        ----------
        voxel : ase.Cell
            An ASE Cell object characterizing the voxel.

        Returns
        -------
        optimal_sigma : float
            The optimal sigma value.
        """
        return (np.max(voxel) / reference_grid_spacing_aluminium) * \
               optimal_sigma_aluminium

    def _calculate(self, atoms, outdir, grid_dimensions, **kwargs):
        if self.parameters._configuration["lammps"]:
            return self.__calculate_lammps(atoms, outdir, grid_dimensions,
                                           **kwargs)
        else:
            return self.__calculate_python(atoms, outdir, grid_dimensions,
                                           **kwargs)

    def __calculate_lammps(self, atoms, outdir, grid_dimensions, **kwargs):
        """Perform actual Gaussian descriptor calculation."""
        use_fp64 = kwargs.get("use_fp64", False)
        return_directly = kwargs.get("return_directly", False)

        lammps_format = "lammps-data"
        ase_out_path = os.path.join(outdir, "lammps_input.tmp")
        ase.io.write(ase_out_path, atoms, format=lammps_format)

        nx = grid_dimensions[0]
        ny = grid_dimensions[1]
        nz = grid_dimensions[2]

        # Check if we have to determine the optimal sigma value.
        if self.parameters.atomic_density_sigma is None:
            self.grid_dimensions = [nx, ny, nz]
            voxel = atoms.cell.copy()
            voxel[0] = voxel[0] / (self.grid_dimensions[0])
            voxel[1] = voxel[1] / (self.grid_dimensions[1])
            voxel[2] = voxel[2] / (self.grid_dimensions[2])
            self.parameters.atomic_density_sigma = self.\
                get_optimal_sigma(voxel)

        # Create LAMMPS instance.
        lammps_dict = {}
        lammps_dict["sigma"] = self.parameters.atomic_density_sigma
        lammps_dict["rcutfac"] = self.parameters.atomic_density_cutoff
        lammps_dict["atom_config_fname"] = ase_out_path
        lmp = self._setup_lammps(nx, ny, nz, outdir, lammps_dict,
                                 log_file_name="lammps_ggrid_log.tmp")

        # For now the file is chosen automatically, because this is used
        # mostly under the hood anyway.
        filepath = __file__.split("atomic_density")[0]
        if self.parameters._configuration["mpi"]:
            if self.parameters.use_z_splitting:
                runfile = os.path.join(filepath, "in.ggrid.python")
            else:
                runfile = os.path.join(filepath, "in.ggrid_defaultproc.python")
        else:
            runfile = os.path.join(filepath, "in.ggrid_defaultproc.python")
        lmp.file(runfile)

        # Extract the data.
        nrows_ggrid = extract_compute_np(lmp, "ggrid",
                                         lammps_constants.LMP_STYLE_LOCAL,
                                         lammps_constants.LMP_SIZE_ROWS)
        ncols_ggrid = extract_compute_np(lmp, "ggrid",
                                         lammps_constants.LMP_STYLE_LOCAL,
                                         lammps_constants.LMP_SIZE_COLS)

        gaussian_descriptors_np = \
            extract_compute_np(lmp, "ggrid",
                               lammps_constants.LMP_STYLE_LOCAL, 2,
                               array_shape=(nrows_ggrid, ncols_ggrid),
                               use_fp64=use_fp64)
        lmp.close()

        # In comparison to SNAP, the atomic density always returns
        # in the "local mode". Thus we have to make some slight adjustments
        # if we operate without MPI.
        self.grid_dimensions = [nx, ny, nz]
        if self.parameters._configuration["mpi"]:
            if return_directly:
                return gaussian_descriptors_np
            else:
                self.fingerprint_length = 4
                return gaussian_descriptors_np, nrows_ggrid
        else:
            # Since the atomic density may be directly fed back into QE
            # during the total energy calculation, we may have to return
            # the descriptors, even in serial mode, without any further
            # reordering.
            if return_directly:
                return gaussian_descriptors_np
            else:
                # Here, we want to do something else with the atomic density,
                # and thus have to properly reorder it.
                # We have to switch from x fastest to z fastest reordering.
                gaussian_descriptors_np = \
                    gaussian_descriptors_np.reshape((grid_dimensions[2],
                                                     grid_dimensions[1],
                                                     grid_dimensions[0],
                                                     7))
                gaussian_descriptors_np = \
                    gaussian_descriptors_np.transpose([2, 1, 0, 3])
                if self.parameters.descriptors_contain_xyz:
                    self.fingerprint_length = 4
                    return gaussian_descriptors_np[:, :, :, 3:], \
                           nx*ny*nz
                else:
                    self.fingerprint_length = 1
                    return gaussian_descriptors_np[:, :, :, 6:], \
                           nx*ny*nz

    def __calculate_python(self, atoms, outdir, grid_dimensions, **kwargs):
        voxel = atoms.cell.copy()
        voxel[0] = voxel[0] / (grid_dimensions[0])
        voxel[1] = voxel[1] / (grid_dimensions[1])
        voxel[2] = voxel[2] / (grid_dimensions[2])
        # gaussian_descriptors_np = np.zeros([np.product(grid_dimensions), 4])
        gaussian_descriptors_np = np.zeros((grid_dimensions[0],
                                             grid_dimensions[1],
                                             grid_dimensions[2], 4),
                                           dtype=np.float64)

        # Hyperparameters
        if self.parameters.atomic_density_sigma is None:
            self.parameters.atomic_density_sigma = self.\
                get_optimal_sigma(voxel)
        cutoff_squared = self.parameters.atomic_density_cutoff*\
                         self.parameters.atomic_density_cutoff
        prefactor = 1.0 /(np.power(self.parameters.atomic_density_sigma*np.sqrt(2*np.pi),3))
        argumentfactor = 1.0 / (2.0 * self.parameters.atomic_density_sigma*
                                self.parameters.atomic_density_sigma)
        print(prefactor,argumentfactor)
        for k in range(0, grid_dimensions[2]):
            for j in range(0, grid_dimensions[1]):
                for i in range(0, grid_dimensions[0]):
                    # Compute the grid.
                    # Orthorhombic cells and triclinic ones have
                    # to be treated differently, see domain.cpp

                    if atoms.cell.orthorhombic:
                        gaussian_descriptors_np[i, j, k, 0:3] = \
                            np.diag(voxel) * [i, j, k]
                    else:
                        gaussian_descriptors_np[i, j, k, 0] = \
                            i/grid_dimensions[0]*atoms.cell[0, 0] + \
                            j/grid_dimensions[1]*atoms.cell[1, 0] + \
                            k/grid_dimensions[2]*atoms.cell[2, 0]

                        gaussian_descriptors_np[i, j, k, 1] = \
                            j/grid_dimensions[1] * atoms.cell[1, 1] + \
                            k/grid_dimensions[2] * atoms.cell[1, 2]

                        gaussian_descriptors_np[i, j, k, 2] = \
                            k/grid_dimensions[2] * atoms.cell[2, 2]

                    # Compute the Gaussians.
                    # Construct a neighborlist for each grid point.
                    neighborlist = ase.neighborlist.NeighborList(
                        np.zeros(len(atoms)+1) +
                        [self.parameters.atomic_density_cutoff],
                        bothways=True,
                    self_interaction=False)

                    atoms_with_grid_point = atoms.copy()
                    atoms_with_grid_point.append(ase.Atom("H",
                                                          gaussian_descriptors_np[i, j, k, 0:3]))
                    neighborlist.update(atoms_with_grid_point)
                    indices, offsets = neighborlist.get_neighbors(len(atoms))
                    nogrid = np.argwhere(indices<len(atoms))
                    indices_nogrid = indices()
                    positions = atoms.get_positions()
                    for a in range(0, len(atoms)):
                        distance_squared = \
                            np.sum(positions[a] -
                                   gaussian_descriptors_np[i, j, k, 0:3])
                        if distance_squared < cutoff_squared:
                            gaussian_descriptors_np[i, j, k, 3] += \
                                prefactor*np.exp(-distance_squared*argumentfactor)

        return gaussian_descriptors_np

