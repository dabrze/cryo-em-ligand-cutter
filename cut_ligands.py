import os
import gc
import logging
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm  # type: ignore
from joblib import Parallel, delayed  # type: ignore
from utils.cryoem_utils import (
    create_histograms,
    extract_ligand_coords,
    read_map,
    get_ligand_mask,
    get_mask_bounding_box,
    get_blob_volume,
    get_sphere_volume,
    resample_blob,
    MAP_VALUE_MAPPER,
)


def extract_ligand(
    output_folder,
    disable_thresholding,
    density_threshold,
    min_blob_radius,
    atom_radius,
    target_voxel_size,
    resolution,
    res_cov_threshold,
    blob_cov_threshold,
    padding,
    unit_cell,
    map_array,
    origin,
    ligand_name,
    ligand_coords,
):
    """
    Extracts a ligand blob from a given map array and saves it as a compressed numpy file.

    Args:
        output_folder (str): Path to the output folder where the compressed numpy file will be saved.
        density_threshold (float): Density threshold for the ligand blob.
        min_blob_radius (float): Minimum radius of the ligand blob.
        atom_radius (float): Radius of the atoms in the ligand.
        target_voxel_size (float): Target voxel size for the resampled blob.
        resolution (float): Resolution of the map.
        res_cov_threshold (float): Minimum coverage threshold for the model.
        blob_cov_threshold (float): Minimum coverage threshold for the blob.
        padding (int): Padding size for the blob.
        unit_cell (np.ndarray): Unit cell dimensions.
        map_array (np.ndarray): Map array.
        origin (np.ndarray): Origin of the map.
        ligand_name (str): Name of the ligand.
        ligand_coords (np.ndarray): Coordinates of the ligand.

    Returns:
        None
    """
    logging.info(f"Cutting out the blob for: {ligand_name}")
    mask = get_ligand_mask(atom_radius, unit_cell, map_array, origin, ligand_coords)
    min_x, max_x, min_y, max_y, min_z, max_z = get_mask_bounding_box(mask)
    blob = map_array * mask
    blob = blob[
        min_x - padding : max_x + 1 + padding,
        min_y - padding : max_y + 1 + padding,
        min_z - padding : max_z + 1 + padding,
    ]

    # Resampling the map to target voxel size
    blob = resample_blob(blob, target_voxel_size, unit_cell, map_array)
    if not disable_thresholding:
        blob[blob < density_threshold] = 0
    else:
        blob[blob <= 0] = 0
    blob_volume = get_blob_volume(np.sum(blob != 0), target_voxel_size)

    if blob_volume >= get_sphere_volume(min_blob_radius):
        fragment_mask = mask[
            min_x - padding : max_x + 1 + padding,
            min_y - padding : max_y + 1 + padding,
            min_z - padding : max_z + 1 + padding,
        ]
        fragment_mask = resample_blob(
            fragment_mask, target_voxel_size, unit_cell, map_array
        )
        res_voxels = fragment_mask > 0
        blob_voxels = blob > 0
        res_cov_frac = np.sum(res_voxels & blob_voxels) / np.sum(res_voxels)
        blob_cov_frac = np.sum(res_voxels & blob_voxels) / np.sum(blob_voxels)

        if res_cov_frac >= res_cov_threshold and blob_cov_frac >= blob_cov_threshold:
            positive_values = blob[blob > 0]
            if disable_thresholding:
                scale_reference = density_threshold
            else:
                scale_reference = positive_values.min()
            scale_reference = max(scale_reference, np.finfo(positive_values.dtype).eps)
            logging.info(
                f"Rescaling min density to: {MAP_VALUE_MAPPER[resolution]} "
                f"(reference: {scale_reference:.6f})"
            )
            blob = blob * (MAP_VALUE_MAPPER[resolution] / scale_reference)
            blob_filename = f"{ligand_name}.npz"

            logging.info(
                f"{ligand_name} Dimensions: {blob.shape}, Blob min value: {blob[blob > 0].min():.3f}, "
                + f"Blob max value: {blob.max():.3f}, Non-zero: {np.sum(blob != 0):,}, "
                + f"Zero: {np.sum(blob == 0):,}, NA count: {np.sum(np.isnan(blob)):,}, "
                + f"Blob volume: {blob_volume:.3f}, Model coverage: {res_cov_frac:.2f}, "
                + f"Blob coverage: {blob_cov_frac:.2f}"
            )
            logging.info(f"Saving blob to: {blob_filename}")
            np.savez_compressed(f"{output_folder}/{blob_filename}", blob)
        else:
            logging.info(
                f"{ligand_name} Model coverage: {res_cov_frac:.2f}, "
                + f"Blob coverage: {blob_cov_frac:.2f}. Not enough coverage. Skipping..."
            )
    else:
        logging.info(f"{ligand_name} Not enough density. Skipping...")


def process_deposit(
    pdb_id,
    input_folder="data",
    output_folder="blobs",
    n_jobs=-1,
    density_std_threshold=2.8,
    min_blob_radius=0.8,
    atom_radius=1.5,
    target_voxel_size=0.2,
    res_cov_threshold=0.02,
    blob_cov_threshold=0.01,
    padding=2,
    verbose=False,
    disable_thresholding=False,
):
    """
    Extracts ligands from a PDB file and saves nearby atom counts to a CSV file.
    Then, reads a map from a CCP4 file and extracts blobs of electron density around each ligand.
    The extracted blobs are saved as separate npz files.

    Args:
        pdb_id (str): The PDB ID of the structure to process.
        input_folder (str, optional): The folder containing the input files. Defaults to "data".
        output_folder (str, optional): The folder to save the output files. Defaults to "blobs".
        n_jobs (int, optional): The number of parallel jobs to run. Defaults to -1.
        density_std_threshold (float, optional): The number of standard deviations from the mean density to
            use as a threshold for blob extraction. Defaults to 2.8.
        min_blob_radius (float, optional): The minimum radius of a blob in angstroms. Defaults to 0.8.
        atom_radius (float, optional): The radius of an atom in angstroms. Defaults to 1.5.
        target_voxel_size (float, optional): The target voxel size in angstroms. Defaults to 0.2.
        res_cov_threshold (float, optional): The minimum ratio of covered voxels to total voxels in a blob for
            it to be considered valid. Defaults to 0.02.
        blob_cov_threshold (float, optional): The minimum ratio of covered voxels to total voxels in a ligand
            for it to be considered valid.
            Defaults to 0.01.
        padding (int, optional): The number of voxels to pad around each ligand. Defaults to 2.
        verbose (bool, optional): Whether to output verbose logging. Defaults to False.
        disable_thresholding (bool, optional): Skip density cutoff and keep raw densities.
    """
    try:
        logging.info("------------------------")
        logging.info(f"Extracting ligands from: {pdb_id}")
        logging.info("------------------------")
        ligands, nearby_noc, resolution, num_particles = extract_ligand_coords(
            f"{input_folder}/{pdb_id}/{pdb_id}.cif"
        )
        if resolution is None:
            logging.warning(f"No resolution found for {pdb_id}. Setting it to 3.0")
            resolution = 3.0
        else:
            logging.info(f"Resolution: {resolution:.1f}")
        if num_particles is None:
            logging.warning(f"Particle num not found for {pdb_id}. Setting it to 3.0")
            resolution = 3.0
        else:
            logging.info(f"Particle num: {num_particles:d}")

        if not ligands:
            logging.info(f"No (studied) ligands found in {pdb_id}. Skipping...")
        else:
            logging.info("Ligands found. Saving nearby atom counts to csv...")
            if not os.path.exists(output_folder):
                logging.info(f"Creating output folder: {output_folder}")
                os.makedirs(output_folder)
            pd.DataFrame.from_dict(
                nearby_noc,
                orient="index",
                columns=[
                    "local_near_cut_count_N",
                    "local_near_cut_count_O",
                    "local_near_cut_count_C",
                ],
            ).to_csv(f"{output_folder}/{pdb_id}_nears.csv")

            logging.info(f"Reading map from: {pdb_id}/map_model_difference_1.ccp4")
            unit_cell, map_array, origin = read_map(
                f"{input_folder}/{pdb_id}/map_model_difference_1.ccp4"
            )
            if origin[0] != 0 or origin[1] != 0 or origin[2] != 0:
                logging.warning(f"Exotic origin: {origin}")

            map_median = np.median(map_array)
            map_std = np.std(map_array)
            value_mask = (map_array < map_median - 0.5 * map_std) | (
                map_array > map_median + 0.5 * map_std
            )
            zeros_pct = np.sum(~value_mask) / map_array.size
            logging.info(f"Median: {map_median:.3f}, std: {map_std:.3f}")
            logging.info(f"Percentage of removed values: {zeros_pct * 100:.2f}%")

            quantile_threshold = norm.cdf(density_std_threshold)
            density_threshold = np.quantile(map_array[value_mask], quantile_threshold)
            logging.info("Quantile threshold: %.5f", quantile_threshold)
            logging.info("Absolute density threshold [V]: %.3f", density_threshold)

            if verbose:
                create_histograms(pdb_id, map_array, value_mask)

            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(extract_ligand)(
                    output_folder,
                    disable_thresholding,
                    density_threshold,
                    min_blob_radius,
                    atom_radius,
                    target_voxel_size,
                    resolution,
                    res_cov_threshold,
                    blob_cov_threshold,
                    padding,
                    unit_cell,
                    map_array,
                    origin,
                    ligand_name,
                    ligand_coords,
                )
                for ligand_name, ligand_coords in ligands.items()
            )
    except Exception as ex:
        logging.error(ex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        help="Directory containing pdb subirectories",
        default="data",
    )
    parser.add_argument(
        "-o", "--output_dir", help="Output directory", required=True, default="blobs"
    )
    parser.add_argument(
        "-p", "--pdb_ids_file", help="File with PDB ids to process", required=True
    )
    parser.add_argument(
        "-n",
        "--n_jobs",
        help="The number of threads to use for a single map. -1 means using all processors. ",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "-l", "--log_file", help="Log filename", default="blob_processing.log"
    )
    parser.add_argument(
        "-d",
        "--density_std_threshold",
        help="Number of std deviations from the mean to use for density threshold (default: 2.8).",
        default=2.8,
        type=float,
    )
    parser.add_argument(
        "--disable_thresholding",
        action="store_true",
        help="Keep raw densities (no quantile thresholding).",
    )
    args = parser.parse_args()

    if args.density_std_threshold == 0:
        raise ValueError("Density std threshold cannot be zero. Set it near zero.")

    logging.basicConfig(
        # filename=args.log_file,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %I:%M:%S",
    )

    logging.info("========================")
    logging.info("Starting blob processing batch.")
    logging.info("========================")

    logging.info(f"Reading input file: {args.pdb_ids_file}")
    with open(args.pdb_ids_file) as file:
        pdb_ids = [line.rstrip() for line in file]
    logging.info(f"Found {len(pdb_ids)} PDB ids in the input file.")

    for pdb_id in pdb_ids:
        process_deposit(
            pdb_id.upper(),
            args.input_dir,
            args.output_dir,
            args.n_jobs,
            args.density_std_threshold,
            disable_thresholding=args.disable_thresholding,
        )
        gc.collect()

    logging.info("========================")
    logging.info("Done.")
    logging.info("========================")
