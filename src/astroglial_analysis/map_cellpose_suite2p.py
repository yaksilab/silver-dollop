import numpy as np


def create_suite2p_cellpose_roi_mapping(cellpose_masks, suite2p_folder):

    suite2p_stat = np.load(f"{suite2p_folder}/stat.npy", allow_pickle=True)

    total_rois = len(suite2p_stat)
    suite2p_roi_indices = np.arange(total_rois)

    # Reconstruct Suite2p masks
    suite2p_masks = np.zeros_like(cellpose_masks, dtype=np.int32)
    for idx in suite2p_roi_indices:
        ypix = suite2p_stat[idx]["ypix"].astype(int)
        xpix = suite2p_stat[idx]["xpix"].astype(int)
        suite2p_masks[ypix, xpix] = idx + 1  # Labels start from 1

    mapping = {}
    suite2p_labels = np.unique(suite2p_masks)
    suite2p_labels = suite2p_labels[suite2p_labels != 0]

    for s2p_label in suite2p_labels:
        s2p_mask = suite2p_masks == s2p_label
        overlapping_labels = cellpose_masks[s2p_mask]
        overlapping_labels = overlapping_labels[overlapping_labels != 0]
        if overlapping_labels.size > 0:
            unique_labels, counts = np.unique(overlapping_labels, return_counts=True)
            most_common_label = unique_labels[np.argmax(counts)]
            mapping[s2p_label] = most_common_label
        else:
            mapping[s2p_label] = None  # No overlap

    return mapping


def map_trace(F, mapping):
    """
    Maps Suite2p traces to Cellpose labels based on a provided mapping and returns a (N, M) NumPy matrix.

    Parameters:
    F (list or numpy.ndarray): A list or array of fluorescence traces from Suite2p.
    mapping (dict): A dictionary where keys are Suite2p ROI labels (1-based index)
                    and values are corresponding Cellpose labels. If a Suite2p ROI
                    does not map to any Cellpose label, the value should be None.

    Returns:
    numpy.ndarray: An (N, M) matrix where the first column is the Cellpose label
                   and the remaining columns are the corresponding trace data.

    """

    mapped_traces = []

    for s2p_label, cp_label in mapping.items():
        s2p_idx = s2p_label - 1  # Adjust index since labels start from 1
        try:
            trace = F[s2p_idx]
        except IndexError:
            print(
                f"Suite2p ROI label {s2p_label} is out of bounds for the provided traces."
            )
            continue

        if cp_label is not None:
            # Prepend the Cellpose label to the trace
            mapped_trace = [cp_label] + list(trace)
            mapped_traces.append(mapped_trace)
        else:
            print(f"No Cellpose mask found for Suite2p ROI {s2p_label}")

    if not mapped_traces:
        print("No traces were mapped to any Cellpose labels.")
        return np.array([])

    first_element = mapped_traces[0][0]

    matrix = np.array(mapped_traces, dtype=type(first_element))

    return matrix
