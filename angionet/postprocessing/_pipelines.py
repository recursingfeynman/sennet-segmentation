import cc3d
import numpy as np


def postprocess(volume: np.ndarray, threshold: int, connectivity: int) -> np.ndarray:
    """
    Apply postprocessing steps for stacked predicted masks
    
    Parameters
    ----------
    volume : np.array
        Stacked predicted masks
    threshold : int
        An integer value used as the threshold for connected components. 
        Connected components with sizes less than this threshold will be removed.
    connectivity : int 
        An integer specifying the connectivity of the connected components. 
        It determines which neighboring pixels are considered connected. 
        Only 4,8 (2D) and 26, 18, and 6 (3D) are allowed.
    
    """
    volume = cc3d.dust(volume, connectivity = connectivity, threshold = threshold)
    return volume