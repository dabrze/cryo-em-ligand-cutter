import os
import logging
import argparse
import numpy as np
import pandas as pd

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
        ligands, nearby_noc = extract_ligand_coords(
            f"{input_folder}/{pdb_id}/{pdb_id}.cif"
        )

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
        default=1,
        type=int,
    )
    parser.add_argument(
        "-l", "--log_file", help="Log filename", default="blob_processing.log"
    )
    args = parser.parse_args()
    logging.basicConfig(
        filename=args.log_file,
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
        process_deposit(pdb_id.upper(), args.input_dir, args.output_dir, args.n_jobs)

    logging.info("========================")
    logging.info("Done.")
    logging.info("========================")
