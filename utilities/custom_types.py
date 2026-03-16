type CoordinatesXYXY = tuple[int, int, int, int]
type CoordinatesNormXYXY = tuple[float, float, float, float]
type ROIObject =  dict[str, CoordinatesNormXYXY] | dict[str, CoordinatesXYXY]
type ROICollection = dict[str, ROIObject]