import mrcfile  # type: ignore
import scipy as sp  # type: ignore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import mode  # type: ignore

from Bio.PDB.MMCIFParser import MMCIFParser  # type: ignore
from scipy import signal
from math import sqrt


def read_map(map_filename):
    """
    Reads a map file in MRC format and returns the unit cell, map array, and origin.

    Args:
        map_filename (str): The path to the MRC file.

    Returns:
        tuple: A tuple containing the unit cell (numpy array), map array (numpy array), and origin (list).
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
    Creates a binary kernel of a given radius.

    Args:
        radius (int): The radius of the kernel.

    Returns:
        numpy.ndarray: A binary kernel of shape (2*radius+1, 2*radius+1, 2*radius+1).
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
    """
    Parses a CIF file and returns the resolution and number of particles
    for the EM reconstruction.

    Args:
        cif_file (str): Path to the input CIF file.

    Returns:
        tuple: A tuple containing the resolution (float) and number of particles (int).
    """
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
    Extracts the coordinates of ligands and nearby atoms from a CIF file.

    Args:
        cif_file (str): The path to the CIF file.

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary of ligand names and their corresponding coordinates.
            - dict: A dictionary of ligand names and their nearby atoms.
            - float: The resolution of the structure.
            - int: The number of particles in the structure.
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
    """
    Determines whether a given residue is a studied ligand.

    Args:
        residue (Bio.PDB.Residue): The residue to check.

    Returns:
        bool: True if the residue is a studied ligand, False otherwise.
    """
    return residue.get_id()[0].startswith("H_")


def find_nearby_NOC_atoms(chain, ligand_residue, ligand_coords, search_radius=3.8):
    """
    Finds nearby nitrogen, oxygen, and carbon atoms to a given ligand residue within a specified search radius.

    Args:
        chain (Bio.PDB.Chain.Chain): The chain containing the ligand residue and other residues to search.
        ligand_residue (Bio.PDB.Residue.Residue): The ligand residue to search around.
        ligand_coords (list): A list of coordinates of atoms in the ligand residue.
        search_radius (float, optional): The search radius in Angstroms. Defaults to 3.8.

    Returns:
        list: A list containing the number of nearby nitrogen, oxygen, and carbon atoms, respectively.
    """

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
    """
    Creates a binary mask of the ligand in the given map_array.

    Args:
        atom_radius (float): The radius of the atoms in the ligand.
        unit_cell (tuple): The dimensions of the unit cell.
        map_array (numpy.ndarray): The 3D density map.
        origin (tuple): The origin of the map.
        ligand_coords (list): The coordinates of the atoms in the ligand.

    Returns:
        numpy.ndarray: A binary mask of the ligand in the given map_array.
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
    Returns the bounding box of a masked array, defined as the minimum and maximum indices of nonzero elements
    in each dimension.

    Args:
        masked_array (numpy.ndarray): A 3D numpy array with boolean values indicating the masked voxels.

    Returns:
        tuple: A tuple containing the minimum and maximum indices of nonzero elements in each dimension, in the
        following order: (min_x, max_x, min_y, max_y, min_z, max_z).
    """
    nonzero_indices = np.nonzero(masked_array)

    min_x, max_x = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])
    min_y, max_y = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])
    min_z, max_z = np.min(nonzero_indices[2]), np.max(nonzero_indices[2])

    return min_x, max_x, min_y, max_y, min_z, max_z


def resample_blob(blob, target_voxel_size, unit_cell, map_array):
    """
    Resamples a given blob to a target voxel size using the provided unit cell and map array.

    Args:
        blob (numpy.ndarray): The blob to be resampled.
        target_voxel_size (float): The target voxel size (in Angstroms).
        unit_cell (tuple): The unit cell dimensions (in Angstroms).
        map_array (numpy.ndarray): The map array.

    Returns:
        numpy.ndarray: The resampled blob.
    """
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
    Calculates the volume of a sphere given its radius.

    Args:
        radius (float): The radius of the sphere.

    Returns:
        float: The volume of the sphere.
    """
    return 4.0 / 3.0 * 3.14 * (radius**3)


def get_blob_volume(voxel_count, voxel_size):
    """
    Calculates the volume of a blob given the number of voxels and the size of each voxel.

    Args:
        voxel_count (int): The number of voxels in the blob.
        voxel_size (float): The size of each voxel in angstroms.

    Returns:
        float: The volume of the blob in cubic angstroms.
    """
    return voxel_count * (voxel_size**3)


def plot_density(blob_array):
    """
    Creates a simple scatter plot of a grid of voxels.

    Parameters:
        blob_array (numpy.ndarray): A 3D array representing the density map.

    Returns:
        None
    """
    z, x, y = blob_array.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c=blob_array[z, x, y])
    plt.show()


def create_histograms(pdb_id, map_array, value_mask):
    """
    Create histograms of map_array values for a given PDB ID.

    Parameters:
        pdb_id (str): The PDB ID of the map.
        map_array (numpy.ndarray): The 3D map array.
        value_mask (numpy.ndarray): A boolean mask indicating which values to include in the histograms.

    Returns:
        None
    """
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


def create_ligand_only_pdb(cif_file, output_file, chimera_dir):
    """
    Creates a PDB file containing only the ligand from a CIF file using Chimera software.

    Parameters:
        cif_file (str): path to the input CIF file
        output_file (str): path to the output PDB file
        chimera_dir (str): path to the Chimera software directory

    Returns:
        None
    """

    chimera_command = f"""
    from chimera import runCommand as rc
    rc('open {cif_file}')
    rc('select :/isHet')
    rc('write selected format pdb {output_file}')
    rc('close all')
    rc('stop now')
    """

    with open("chimera_command.py", "w") as f:
        f.write(chimera_command)

    os.system(f"{chimera_dir}/bin/chimera --nogui --script chimera_command.py")
    os.remove("chimera_command.py")


def calc_qscores(map_file, pdb_file, mapq_dir, chimera_dir, np=6, res=3.0, sigma=0.6):
    """
    Calculates the Q-scores for a given map and PDB file using the MAPQ tool.

    Parameters:
        map_file (str): Path to the map file.
        pdb_file (str): Path to the PDB file.
        mapq_dir (str): Path to the MAPQ directory.
        chimera_dir (str): Path to the Chimera directory.
        np (int, optional): Number of processors to use. Defaults to 6.
        res (float, optional): Reference resolution of the Q-scores. Defaults to 3.0.
        sigma (float, optional): Sigma value for the map. Defaults to 0.6.

    Returns:
        Tuple[str, str]: A tuple containing the paths to the resulting PDB file and
        the text file with all the Q-scores.
    """

    pdb_result_file = f"{pdb_file}__Q__{map_file}.pdb"
    txt_result_file = f"{pdb_file}__Q__{map_file}_All.txt"

    os.system(
        f"{mapq_dir}/mapq_cmd.py {chimera_dir} {map_file} {pdb_file} np={np} res={res} sigma={sigma}"
    )

    return pdb_result_file, txt_result_file


def extract_qscores(qscores_txt_file):
    parsing = False
    qscore_dict = {"chain": [], "res_id": [], "res_name": [], "qscore": []}

    with open(qscores_txt_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("Chain	Molecule"):
            parsing = True
            continue

        if line.startswith("Molecule"):
            parsing = False

        if parsing:
            ligand_stats = line.split()
            if len(ligand_stats) < 4:
                continue
            qscore_dict["chain"].append(ligand_stats[0])
            qscore_dict["res_name"].append(ligand_stats[1])
            qscore_dict["res_id"].append(ligand_stats[2])
            qscore_dict["qscore"].append(float(ligand_stats[3]))

    return pd.DataFrame(qscore_dict)


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
