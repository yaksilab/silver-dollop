
import numpy as np

def map_cellpose_suite2p_all_rois(cellpose_mask_path, suite2p_folder):

    cellpose_masks = np.load(cellpose_mask_path, allow_pickle=True).item()['masks']

    suite2p_stat = np.load(f'{suite2p_folder}/stat.npy', allow_pickle=True)

    total_rois = len(suite2p_stat)
    suite2p_roi_indices = np.arange(total_rois)

    # Reconstruct Suite2p masks
    suite2p_masks = np.zeros_like(cellpose_masks, dtype=np.int32)
    for idx in suite2p_roi_indices:
        ypix = suite2p_stat[idx]['ypix'].astype(int)
        xpix = suite2p_stat[idx]['xpix'].astype(int)
        suite2p_masks[ypix, xpix] = idx + 1  # Labels start from 1

    # Map masks
    mapping = {}
    suite2p_labels = np.unique(suite2p_masks)
    suite2p_labels = suite2p_labels[suite2p_labels != 0]

    for s2p_label in suite2p_labels:
        s2p_mask = (suite2p_masks == s2p_label)
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
    cellpose_traces = {}

    for s2p_label, cp_label in mapping.items():
        s2p_idx = s2p_label - 1  # Adjust index since labels start from 1
        trace = F[s2p_idx]
        if cp_label is not None:
            # Assign the trace to the Cellpose label
            if cp_label in cellpose_traces:
                # If multiple Suite2p ROIs map to the same Cellpose mask,store them in a list
                cellpose_traces[cp_label].append(trace)
            else:
                cellpose_traces[cp_label] = [trace]
        else:
            print(f"No Cellpose mask found for Suite2p ROI {s2p_label}")
            pass  # You can choose to store these separately if needed

    return cellpose_traces
