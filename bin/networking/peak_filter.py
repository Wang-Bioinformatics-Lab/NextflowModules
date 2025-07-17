import numpy as np
class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = np.zeros(size + 1, dtype=np.float64)

    def add(self, position):
        # position = self.size - position - 1
        while position < self.size:
            self.tree[position] += 1
            position = position | (position + 1)

    def query(self, position):
        sum = 0
        # position = self.size - position - 1
        while (position >= 0):
            sum += self.tree[position]
            position = (position & (position + 1)) - 1
        
        return sum

def filter_window_peaks_optimized(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    window_size: float = 50,
    peaks_to_keep_in_window: int = 6
) -> (np.ndarray, np.ndarray):
    """
    For each peak, calculate the number of peaks within its m/z window 
    (current_mz - window_size, current_mz + window_size) 
    that have larger intensity. Keep only the peaks that have less 
    than `peaks_to_keep_in_window` such peaks in their window.
    Assumes the input mz_array is sorted.
    """
    
    if len(mz_array) == 0:
        return mz_array, intensity_array

    n = len(mz_array)
    
    rev_sorted_intensities = np.argsort(-intensity_array)
    fenwick_tree = FenwickTree(n)
    res = []
    
    for i in range(n):
        current_mz = mz_array[rev_sorted_intensities[i]]
        current_intensity = intensity_array[rev_sorted_intensities[i]]
        lower_mz_bound = current_mz - window_size
        start_idx = np.searchsorted(mz_array, lower_mz_bound, side='left')
        left = fenwick_tree.query(rev_sorted_intensities[i]) - fenwick_tree.query(start_idx)
        end_idx = np.searchsorted(mz_array, current_mz + window_size, side='right')
        right = fenwick_tree.query(end_idx) - fenwick_tree.query(rev_sorted_intensities[i])
        
        if left + right <= peaks_to_keep_in_window:
            res.append((current_mz, current_intensity))
        
        # Update Fenwick tree with the current peak
        fenwick_tree.add(rev_sorted_intensities[i])


    mz_array, intensity_array = zip(*res)
    # sort the results by m/z
    index = np.argsort(mz_array)
    mz_array = np.array(mz_array, dtype=np.float64)[index]
    intensity_array = np.array(intensity_array, dtype=np.float64)[index]
    return np.array(mz_array, dtype=np.float64), np.array(intensity_array, dtype=np.float64)


def smart_filter_window_peaks_optimized(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    window_size: float = 50,
    peaks_to_keep_in_window: int = 6
) -> (np.ndarray, np.ndarray):
    """
    For each peak, calculate the number of peaks within its m/z window 
    (current_mz - window_size, current_mz + window_size) 
    that have larger intensity. Keep only the peaks that have less 
    than `peaks_to_keep_in_window` such peaks in their window.
    Assumes the input mz_array is sorted.
    """
    
    if len(mz_array) == 0:
        return mz_array, intensity_array
    
    #---------------- this is to improve speed, it will cause change in result compared to legacy GNPS ----------------
    mz_array, intensity_array = filter_window_peaks(
        mz_array, intensity_array, window_size=window_size//4, peaks_to_keep_in_window=peaks_to_keep_in_window
    )

    n = len(mz_array)
    # Create a boolean mask to mark peaks to keep
    keep_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        current_mz = mz_array[i]
        current_intensity = intensity_array[i]

        #---------------correct way to do it:----------------
        # count_larger_in_window = np.sum(
        #     (mz_array > (current_mz - window_size)) &
        #     (mz_array < (current_mz + window_size)) &
        #     (intensity_array > current_intensity)
        # )
        
        #----------------gnps legacy way:----------------
        mask = ((mz_array > (current_mz - window_size)) &
            (mz_array < (current_mz + window_size)) &
            (intensity_array > current_intensity))
        #------------------------------------------------
        
        # count unique values in the intensity array:
        count_larger_in_window = len(np.unique(intensity_array[mask]))
        

        if count_larger_in_window < peaks_to_keep_in_window:
            keep_mask[i] = True # Mark this peak to be kept

    return mz_array[keep_mask], intensity_array[keep_mask]

def filter_window_peaks(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    window_size: float = 50,
    peaks_to_keep_in_window: int = 6
) -> (np.ndarray, np.ndarray):
    """
    Sliding‐window peak filter.  For each peak i, consider all peaks j
    with |mz[j] - mz[i]| <= window_size.  Keep peak i if its intensity
    is among the top `peaks_to_keep_in_window` in that local window.

    Returns filtered (mz, intensity), sorted by mz ascending.
    """
    if len(mz_array) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # 1) find the maximum mass to size our buckets
    max_mass = np.max(mz_array)

    # 2) create buckets
    num_buckets = int(int(max_mass) // window_size) + 1
    buckets = [list() for _ in range(num_buckets)]

    # 3) assign each peak to its bucket
    for mass, intensity in zip(mz_array, intensity_array):
        idx = int(int(mass) // window_size)
        buckets[idx].append((mass, intensity))

    # 4) sort each bucket by intensity descending
    for bucket in buckets:
        bucket.sort(key=lambda peak: peak[1], reverse=True)

    # 5) collect up to max_peaks per bucket
    filtered = list()
    for bucket in buckets:
        if len(bucket) > peaks_to_keep_in_window:
            bucket = bucket[:peaks_to_keep_in_window]
        filtered.extend(bucket)

    # 6) sort the final list by mass ascending
    filtered.sort(key=lambda peak: peak[0])
    # 7) convert to numpy arrays
    mz_filtered = np.array([peak[0] for peak in filtered], dtype=np.float64)
    int_filtered = np.array([peak[1] for peak in filtered], dtype=np.float64)
    return mz_filtered, int_filtered


def filter_around_precursor(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    precursor_mz,
    window_size = 17.0
    ) -> (np.ndarray, np.ndarray):
    """
    Filter peaks around the precursor m/z value.
    Peaks within 50 Da of the precursor m/z are kept.
    Returns filtered (mz, intensity), sorted by mz ascending.
    """
    if len(mz_array) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    
    # Create a mask for peaks within the window size of the precursor m/z
    mask = np.abs(mz_array - precursor_mz) >= window_size
    
    mz_filtered = mz_array[mask]
    int_filtered = intensity_array[mask]
    if len(mz_filtered) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    else:
        return mz_filtered.astype(np.float64), int_filtered.astype(np.float64)

def filter_peaks_optimized(mz_array, intensity_array, precursor_mz, precursor_charge,
                           filter_precursor=True, filter_window=True, filter_precursor_radius=17.0,
                           filter_window_size=50.0, peaks_to_keep_in_window=6):
    """Only apply sqrt transform and L2 normalization without filtering"""
    if len(mz_array) == 0:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            precursor_mz,
            precursor_charge
        )

    # Sort by m/z first
    sorted_idx = np.argsort(mz_array)
    mz_sorted = mz_array[sorted_idx].astype(np.float64)
    int_sorted = intensity_array[sorted_idx].astype(np.float64)
    
    if filter_precursor:
        # Filter around precursor m/z
        mz_sorted, int_sorted = filter_around_precursor(
            mz_sorted, int_sorted, precursor_mz, window_size=filter_precursor_radius
        )
    
    if filter_window:
        mz_sorted, int_sorted = smart_filter_window_peaks_optimized(
            mz_sorted, int_sorted,
            window_size=filter_window_size,
            peaks_to_keep_in_window=peaks_to_keep_in_window
        )
    

    int_sorted = np.sqrt(int_sorted)
    # Apply sqrt transform and L2 normalization
    norm = np.linalg.norm(int_sorted)
    if norm > 0:
        int_sorted /= norm

    return (mz_sorted, int_sorted, np.float32(precursor_mz), np.int32(precursor_charge)) # precursor mass is converted to float 32 to match legacy GNPS result.
