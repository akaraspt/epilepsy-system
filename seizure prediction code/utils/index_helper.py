import numpy as np


def get_list_con_seq_idx(input_idx):
    """
    Get a list of continuous sequence indices.

    Parameters
    ----------
    input_idx : numpy array
        Input indices.

    Returns
    -------
    list_sql_idx : numpy array
        A list of continuous sequence indices.

    """

    # If there is no input indices, return none
    if input_idx.size == 0:
        return np.empty(0, dtype=object)

    # Generate a list of continuous sequence indices
    def gen_con_seq_idx(src_idx, end_idx):
        # Get around indexing problem of arange()
        adj_src_idx = src_idx + 1

        # Create ndarray of index
        idx_range = np.empty(adj_src_idx.size + 1, dtype=object)
        for i in range(idx_range.size):
            if i == 0:
                idx_range[i] = np.arange(adj_src_idx[i])
            elif i == idx_range.size - 1:
                idx_range[i] = np.arange(adj_src_idx[i - 1], end_idx)
            else:
                idx_range[i] = np.arange(adj_src_idx[i - 1], adj_src_idx[i])
        return idx_range

    # Get end indices of continuous sequence indices of input indices
    idx_end_input_idx = np.where(np.diff(input_idx) > 1)[0]

    # If there is only one sequence
    if idx_end_input_idx.size == 0 and input_idx.size > 0:
        list_seq_idx = np.empty(1, dtype=object)
        list_seq_idx[0] = np.arange(input_idx.size)
    # If there are more than one sequences
    else:
        list_seq_idx = gen_con_seq_idx(idx_end_input_idx, input_idx.size)

    return list_seq_idx
