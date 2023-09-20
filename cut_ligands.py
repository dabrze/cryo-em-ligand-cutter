import os
import logging
import numpy as np

from joblib import Parallel, delayed
from cryoem_utils import (
    extract_ligand_coords,
    read_map,
    get_ligand_mask,
    get_mask_bounding_box,
    get_blob_volume,
    get_sphere_volume,
    resample_blob,
)

logging.basicConfig(
    filename="blob_processing.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S",
)


def extract_ligand(
    output_folder,
    density_std_threshold,
    min_blob_radius,
    atom_radius,
    target_voxel_size,
    target_min_density,
    padding,
    unit_cell,
    map_array,
    origin,
    ligand_name,
    ligand_coords,
):
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
    blob[blob < density_std_threshold * np.std(map_array)] = 0
    blob_volume = get_blob_volume(np.sum(blob != 0), target_voxel_size)

    if blob_volume >= get_sphere_volume(min_blob_radius):
        # rescaling density values
        blob = blob * (target_min_density / blob[blob > 0].min())
        blob_filename = f"{ligand_name}.npz"

        logging.info(
            f"{ligand_name} Dimensions: {blob.shape}, Blob min value: {blob[blob > 0].min():.3f}, "
            + f"Blob max value: {blob.max():.3f}, Non-zero: {np.sum(blob != 0):,}, "
            + f"Zero: {np.sum(blob == 0):,}, NA count: {np.sum(np.isnan(blob)):,}"
        )
        logging.info(f"Saving blob to: {blob_filename}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        np.savez_compressed(f"{output_folder}/{blob_filename}", blob)
    else:
        logging.info(f"{ligand_name} Not enough density. Skipping...")


def process_deposit(
    pdb_id,
    input_folder="data",
    output_folder="blobs",
    n_jobs=-1,
    density_std_threshold=17,
    min_blob_radius=0.8,
    atom_radius=1.5,
    target_voxel_size=0.2,
    target_min_density=0.39,
    padding=2,
):
    try:
        logging.info("------------------------")
        logging.info(f"Extracting ligands from: {pdb_id}")
        logging.info("------------------------")
        ligands = extract_ligand_coords(f"{input_folder}/{pdb_id}.cif")

        if not ligands:
            logging.info(f"No (studied) ligands found in {pdb_id}. Skipping...")
        else:
            logging.info(f"Reading map from: {pdb_id}")
            unit_cell, map_array, origin = read_map(
                f"{input_folder}/{pdb_id}_map_model_difference_1.ccp4"
            )

            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(extract_ligand)(
                    output_folder,
                    density_std_threshold,
                    min_blob_radius,
                    atom_radius,
                    target_voxel_size,
                    target_min_density,
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
    logging.info("========================")
    logging.info("Starting blob processing batch.")
    logging.info("========================")

    for pdb_id in ["8sor", "6kpj", "8scx"]:
        process_deposit(pdb_id)

    logging.info("Done.")
