import mrcfile
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode

from Bio.PDB.MMCIFParser import MMCIFParser
from scipy import signal
from math import sqrt


def read_map(map_filename):
    """
    Reads a map file and converts it to a voxel grid (array) and adjust the axes order.
    """
    with mrcfile.open(map_filename) as file:
        order = (3 - file.header.maps, 3 - file.header.mapr, 3 - file.header.mapc)
        map_array = np.asarray(file.data, dtype="float")
        map_array = np.moveaxis(a=map_array, source=(0, 1, 2), destination=order)

        unit_cell = np.zeros(6, dtype="float")
        cell = file.header.cella[["x", "y", "z"]]
        unit_cell[:3] = cell.astype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]).view(
            ("<f4", 3)
        )

        # swapping a and c to compatible with ZYX convension
        unit_cell[0], unit_cell[2] = unit_cell[2], unit_cell[0]
        unit_cell[3:] = float(90)
        origin = [
            1 * file.header.nxstart,
            1 * file.header.nystart,
            1 * file.header.nzstart,
        ]

    return unit_cell, map_array, origin


def create_binary_kernel(radius):
    """
    Creates a binary kernel for creating convolutions around atoms.
    """
    boxsize = 2 * radius + 1
    kern_sphere = np.zeros(shape=(boxsize, boxsize, boxsize), dtype="float")
    kx = ky = kz = boxsize
    center = boxsize // 2

    r1 = center
    for i in range(kx):
        for j in range(ky):
            for k in range(kz):
                dist = sqrt((i - center) ** 2 + (j - center) ** 2 + (k - center) ** 2)
                if dist < r1:
                    kern_sphere[i, j, k] = 1

    return kern_sphere


def get_em_stats(cif_file):
    resolution = None
    num_particles = None

    for line in open(cif_file):
        if line.startswith("_em_3d_reconstruction.resolution "):
            try:
                resolution = round(float(line.split()[1]), 1)
            except Exception:
                resolution = None
        if line.startswith("_em_3d_reconstruction.num_particles "):
            try:
                num_particles = int(line.split()[1])
            except Exception:
                num_particles = None

    if resolution is not None:
        if resolution > 4.0:
            resolution = 4.0
        elif resolution < 1.0:
            resolution = 1.0

    return resolution, num_particles


def extract_ligand_coords(cif_file):
    """
    Extracts the coordinates of all the atoms of a ligand in a CIF file.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("cif", cif_file)
    resolution, num_particles = get_em_stats(cif_file)

    pdb_id = cif_file.split("/")[-1][:-4]
    model = structure[0]
    ligands = {}
    ligand_nearby_atoms = {}

    for chain in model:
        chain_id = chain.get_id()

        for residue in chain:
            if is_studied_ligand(residue):
                ligand_coords = []
                ligand_name = f"{pdb_id}_{chain_id}_{residue.get_id()[1]}_{residue.get_id()[0][2:]}"
                for atom in residue:
                    ligand_coords.append(atom.get_coord())

                ligands[ligand_name] = ligand_coords
                ligand_nearby_atoms[ligand_name] = find_nearby_NOC_atoms(
                    chain, residue, ligand_coords
                )

    return ligands, ligand_nearby_atoms, resolution, num_particles


def is_studied_ligand(residue):
    return residue.get_id()[0].startswith("H_")


def find_nearby_NOC_atoms(chain, ligand_residue, ligand_coords, search_radius=3.8):
    nearby_n = 0
    nearby_o = 0
    nearby_c = 0

    for residue in chain:
        if residue != ligand_residue:
            for atom in residue:
                if atom.element in ["N", "O", "C"]:
                    atom_coordinate = atom.get_vector()

                    for ligand_atom_coord in ligand_coords:
                        distance = (atom_coordinate - ligand_atom_coord).norm()
                        if distance <= search_radius:
                            if atom.element == "C":
                                nearby_c += 1
                            elif atom.element == "O":
                                nearby_o += 1
                            else:
                                nearby_n += 1

    return [nearby_n, nearby_o, nearby_c]


def get_ligand_mask(atom_radius, unit_cell, map_array, origin, ligand_coords):
    """Masks the map using the coordinates of ligand atoms.

    Args:
        atom_radius (_type_): _description_
        unit_cell (_type_): _description_
        map_array (_type_): _description_
        origin (_type_): _description_
        ligand_coords (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_ligand = np.array(ligand_coords)[:, 0] - origin[0]
    y_ligand = np.array(ligand_coords)[:, 1] - origin[1]
    z_ligand = np.array(ligand_coords)[:, 2] - origin[2]

    grid_3d = np.zeros((map_array.shape), dtype="float")
    x = x_ligand * map_array.shape[0] / unit_cell[0]
    y = y_ligand * map_array.shape[1] / unit_cell[1]
    z = z_ligand * map_array.shape[2] / unit_cell[2]

    for ix, iy, iz in zip(x, y, z):
        grid_3d[int(round(iz)), int(round(iy)), int(round(ix))] = 1.0

    pixsize = unit_cell[0] / map_array.shape[0]
    kern_rad = round(atom_radius / pixsize)
    if kern_rad > 0:
        grid2 = signal.fftconvolve(grid_3d, create_binary_kernel(kern_rad), "same")
        grid2_binary = grid2 > 1e-5
        dilate = sp.ndimage.morphology.binary_dilation(grid2_binary, iterations=1)
        mask = dilate
    else:
        mask = grid_3d

    mask = mask * (mask >= 1.0e-5)
    mask = np.where(grid2_binary, 1.0, mask)
    shift_z = origin[0]
    shift_y = origin[1]
    shift_x = origin[2]
    mask = np.roll(
        np.roll(np.roll(mask, -shift_z, axis=0), -shift_y, axis=1),
        -shift_x,
        axis=2,
    )

    return mask


def get_mask_bounding_box(masked_array):
    """
    Calculates the 3D bounding box of non-zero voxels in an array.
    """
    nonzero_indices = np.nonzero(masked_array)

    min_x, max_x = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])
    min_y, max_y = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])
    min_z, max_z = np.min(nonzero_indices[2]), np.max(nonzero_indices[2])

    return min_x, max_x, min_y, max_y, min_z, max_z


def resample_blob(blob, target_voxel_size, unit_cell, map_array):
    blob = sp.ndimage.zoom(
        blob,
        [
            unit_cell[0] / target_voxel_size / map_array.shape[0],
            unit_cell[1] / target_voxel_size / map_array.shape[1],
            unit_cell[2] / target_voxel_size / map_array.shape[2],
        ],
        prefilter=False,
    )

    return blob


def get_sphere_volume(radius):
    """
    Calculates the volume of a sphere with a given radius.
    """
    return 4.0 / 3.0 * 3.14 * (radius**3)


def get_blob_volume(voxel_count, voxel_size):
    """
    Calculates the Angstrom volume of a blob based on the number of voxels and their size (in Angstroms).
    """
    return voxel_count * (voxel_size**3)


def plot_density(blob_array):
    """
    Creates a simple scatter plot of a grid of voxels.
    """
    z, x, y = blob_array.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c=blob_array[z, x, y])
    plt.show()


def create_histograms(pdb_id, map_array, value_mask):
    map_mode = mode(map_array.flatten())[0]

    df = pd.DataFrame(map_array.flatten(), columns=[f"{pdb_id} original"])
    df.hist(bins=100)
    plt.savefig(f"hists/{pdb_id}_original.png")
    df = pd.DataFrame(
        map_array[(map_array < map_mode) | (map_array > map_mode)].flatten(),
        columns=[f"{pdb_id} without mode"],
    )
    df.hist(bins=100)
    plt.savefig(f"hists/{pdb_id}_without mode.png")
    df = pd.DataFrame(
        map_array[value_mask].flatten(),
        columns=[f"{pdb_id} without 1/2 std around median"],
    )
    df.hist(bins=100)
    plt.savefig(f"hists/{pdb_id}_without_half_std arounf_median.png")


MAP_VALUE_MAPPER = {
    1.0: 0.66,
    1.1: 0.63,
    1.2: 0.57,
    1.3: 0.57,
    1.4: 0.54,
    1.5: 0.50,
    1.6: 0.48,
    1.7: 0.44,
    1.8: 0.42,
    1.9: 0.39,
    2.0: 0.36,
    2.1: 0.33,
    2.2: 0.31,
    2.3: 0.30,
    2.4: 0.28,
    2.5: 0.25,
    2.6: 0.25,
    2.7: 0.23,
    2.8: 0.21,
    2.9: 0.21,
    3.0: 0.20,
    3.1: 0.18,
    3.2: 0.18,
    3.3: 0.17,
    3.4: 0.15,
    3.5: 0.16,
    3.6: 0.14,
    3.7: 0.12,
    3.8: 0.14,
    3.9: 0.15,
    4.0: 0.17,
}
