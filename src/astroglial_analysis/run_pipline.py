import numpy as np
from astroglial_segmentation import create_suite2p_masks_extract_traces

from .classifier2 import classify_masks, get_ab
from .create_masks import create_cp_mask
from .interpolated_function import (
    get_all_intersections,
    parametrize_curve,
    get_formated_region_coords,
    uniform_align_processes,
)
from .determine_line import get_line, uniform_align_comp_cell
from .sub_segmentation import sub_segment
from scipy.io import savemat
from astroglial_analysis.map_cellpose_suite2p import (
    create_suite2p_cellpose_roi_mapping,
    map_trace,
)
import os


def save_as_mat(matrix, path):
    savemat(path, {"matrix": matrix})
    print(f"Matrix saved to {path} as a .mat file")


def save_as_npy(matrix, path):
    np.save(path, matrix)
    print(f"Matrix saved to {path} as a .npy file")


def get_correspondence_matrix(masks):

    _, _, processes, complete_cells = classify_masks(masks)
    upper_complete, lower_complete = complete_cells["upper"], complete_cells["lower"]
    upper_processes, lower_processes = processes["upper"], processes["lower"]
    ab_list_upper = [
        (up, get_ab(get_formated_region_coords(np.where(masks == up))))
        for up in upper_processes
    ]
    ab_list_lower = [
        (low, get_ab(get_formated_region_coords(np.where(masks == low))))
        for low in lower_processes
    ]
    line_upper, _ = get_line(upper_complete, masks, True, 20)
    line_lower, _ = get_line(lower_complete, masks, False, 20)
    upper_points = np.array([point[0] for point in line_upper])
    lower_points = np.array([point[0] for point in line_lower])
    t_upper, tck_upper, _, _, _ = parametrize_curve(upper_points)
    t_lower, tck_lower, _, _, _ = parametrize_curve(lower_points)

    intersection_list_upper = get_all_intersections(
        tck_upper, t_upper, ab_list_upper, upper=True
    )
    intersection_list_lower = get_all_intersections(
        tck_lower, t_lower, ab_list_lower, upper=False
    )

    _, cor_matrix_upper_comp = uniform_align_comp_cell(line_upper, masks, True)
    _, cor_matrix_lower_comp = uniform_align_comp_cell(line_lower, masks, False)

    _, cor_matrix_upper_p = uniform_align_processes(
        intersection_list_upper, masks, True
    )
    _, cor_matrix_lower_p = uniform_align_processes(
        intersection_list_lower, masks, False
    )

    total_cor_matrix = np.vstack(
        [
            cor_matrix_upper_comp,
            cor_matrix_lower_comp,
            cor_matrix_upper_p,
            cor_matrix_lower_p,
        ]
    )
    return total_cor_matrix


def run_pipeline(working_directory):
    combined_mean_image_seg_path = os.path.join(
        working_directory, "combined_mean_image_seg.npy"
    )
    mask_file = np.load(combined_mean_image_seg_path, allow_pickle=True)
    new_maskfile = mask_file.copy()
    masks = mask_file.item()["masks"]

    total_cor_matrix = get_correspondence_matrix(masks)

    sub_segmented_data = sub_segment(total_cor_matrix, 10)
    subsegmented_mask = create_cp_mask(sub_segmented_data, masks)

    new_maskfile.item()["masks"] = subsegmented_mask
    output_path = os.path.join(working_directory, "subsegmented_masks_seg.npy")
    np.save(output_path, new_maskfile)
    print(f"New mask array saved to {output_path}")

    subsegmented_masks_seg_path = os.path.join(
        working_directory, "subsegmented_masks_seg.npy"
    )

    create_suite2p_masks_extract_traces(working_directory, "subsegmented_masks_seg.npy")

    suite2p_folder = os.path.join(working_directory, "cellpose_suite2p_output")

    mapping = create_suite2p_cellpose_roi_mapping(subsegmented_mask, suite2p_folder)

    traces = np.load(f"{suite2p_folder}/F.npy", allow_pickle=True)
    mapped_traces_matrix = map_trace(traces, mapping)

    npy_output_path_trace = os.path.join(working_directory, "trace_matrix.npy")
    mat_output_path_trace = os.path.join(working_directory, "trace_matrix.mat")
    npy_output_path_correspondence_matrix = os.path.join(
        working_directory, "correspondence_matrix.npy"
    )
    mat_output_path_correspondence_matrix = os.path.join(
        working_directory, "correspondence_matrix.mat"
    )
    save_as_mat(mapped_traces_matrix, mat_output_path_trace)
    save_as_mat(sub_segmented_data, mat_output_path_correspondence_matrix)
    # save_as_npy(mapped_traces_matrix, npy_output_path)
    # save_as_npy(total_cor_matrix, npy_output_path_correspondence_matrix)
