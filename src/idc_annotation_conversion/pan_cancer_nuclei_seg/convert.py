"""Utilities for converting annotations. Clear of cloud-specific things."""
import logging
from typing import Tuple, Union
import highdicom as hd
import numpy as np
import pandas as pd
from pydicom import Dataset
from pydicom.sr.codedict import codes
from shapely.geometry.polygon import Polygon

from idc_annotation_conversion.pan_cancer_nuclei_seg import metadata_config


def process_df_row(
    df_row: pd.Series,
    transformer: hd.spatial.ImageToReferenceTransformer,
    graphic_type: hd.ann.GraphicTypeValues = hd.ann.GraphicTypeValues.POLYGON,
    annotation_coordinate_type: hd.ann.AnnotationCoordinateTypeValues = hd.ann.AnnotationCoordinateTypeValues.SCOORD,  # noqa: E501
) -> Union[Tuple[Polygon, np.ndarray, float], Tuple[None, None, None]]:
    """Process a single annotation.

    Parameters
    ----------
    df_row: pd.Series
        Single row of the dataframe containing the annotations.
    transformer: hd.spatial.ImageToReferenceTransformer
        Transformer object to map image coordinates to reference coordinates
        for the image.
    graphic_type: highdicom.ann.GraphicTypeValues, optional
        Graphic type to use to store all nuclei. Note that all but 'POLYGON'
        result in simplification and loss of information in the annotations.
    annotation_coordinate_type: Union[hd.ann.AnnotationCoordinateTypeValues, str], optional
        Store coordinates in the Bulk Microscopy Bulk Simple Annotations in the
        (3D) frame of reference (SCOORD3D), or the (2D) total pixel matrix
        (SCOORD, default).

    Returns
    -------
    polygon_image: shapely.Polygon
        Polygon (in image coordinates) representing the annotation in the CSV.
        Note that this is always the full original polygon regardless of the
        requested graphic type.
    graphic_data: np.ndarray
        Numpy array of float32 coordinates to include in the Bulk Microscopy
        Simple Annotations.
    identifier: int 
        Identifier for each annotation taken as is from input. 

    """  
    # TODO: 
    points = np.array(
        csv_row.Polygon[1:-1].split(':'),
        dtype=np.float32
    )
    
    n = len(points) // 2
    if points.shape[0] < 6 or (points.shape[0] % 2) == 1:
        return None, None, None
    full_coordinates_image = points.reshape(n, 2)
    polygon_image = Polygon(full_coordinates_image)

    # Remove the final point (require not to be closed but Polygon adds
    # this)
    coords = np.array(polygon_image.exterior.coords)[:-1, :]

    # Remove the last point if it is the same as the first (in this case the
    # duplicate comes from the original CSV file)
    if (coords[0, :] == coords[-1, :]).all():
        coords = coords[:-1, :]
        # There seem to be a small number of cases with only three points, with
        # the first and last duplicated. Just remove these.
        if len(coords) < 3:
            return None, None, None

    # Simplify the coordinates as required
    if graphic_type == hd.ann.GraphicTypeValues.POLYGON:
        graphic_data = coords
    elif graphic_type == hd.ann.GraphicTypeValues.POINT:
        x, y = polygon_image.centroid.xy
        graphic_data = np.array([[x[0], y[0]]])
    elif graphic_type == hd.ann.GraphicTypeValues.RECTANGLE:
        # The rectangle need not be axis aligned but here we
        # do standardize to an axis aligned rectangle
        minx, miny, maxx, maxy = polygon_image.bounds
        graphic_data = np.array(
            [
                [minx, miny],
                [maxx, miny],
                [maxx, maxy],
                [minx, maxy],
            ]
        )
    elif graphic_type == hd.ann.GraphicTypeValues.ELLIPSE:
        # Find the minimum rotated rectangle that includes all points in the
        # polygon, then use the midpoints of these lines as the endpoints of
        # the major and minor axes of the ellipse. This is a convenient, if
        # somewhat crude way of approximating the polygon with an ellipse.
        # Note that the resulting ellipse will not in general contain all the
        # points of the original polygon, some may be outside
        rec = np.array(polygon_image.minimum_rotated_rectangle.exterior.coords)

        # Array of midpoints
        graphic_data = np.array(
            [
                (rec[0] + rec[1]) / 2,
                (rec[2] + rec[3]) / 2,
                (rec[1] + rec[2]) / 2,
                (rec[0] + rec[3]) / 2,
            ]
        )
        # Ensure we have the major axis endpoints first
        d1 = np.linalg.norm(graphic_data[1] - graphic_data[0])
        d2 = np.linalg.norm(graphic_data[3] - graphic_data[2])
        if d2 > d1:
            # Swap first two points with second two points
            graphic_data = graphic_data[[2, 3, 0, 1], :]
    else:
        raise ValueError(
            f'Graphic type '{graphic_type.value}' not supported.'
        )

    use_3d = (
        annotation_coordinate_type ==
        hd.ann.AnnotationCoordinateTypeValues.SCOORD3D
    )
    if use_3d:
        graphic_data = transformer(graphic_data)

    return polygon_image, graphic_data.astype(np.float32), df_row['cell_id']


def get_graphic_data(
    annotations: pd.DataFrame,
    source_image_metadata: Dataset,
    graphic_type: Union[
        hd.ann.GraphicTypeValues,
        str
    ] = hd.ann.GraphicTypeValues.POLYGON,
    annotation_coordinate_type: Union[
        hd.ann.AnnotationCoordinateTypeValues,
        str
    ] = hd.ann.AnnotationCoordinateTypeValues.SCOORD,
) -> tuple[list[Polygon], list[np.ndarray], list[float]]:
    """Parse annotations to construct graphic data.

    Parameters
    ----------
    annotations: pd.DataFrame
        Dataframe containing at least following columns
        # TODO 
    source_image_metadata: pydicom.Dataset
        Pydicom datasets containing the metadata of the image (already
        converted to DICOM format). Note that this should be the metadata of
        the image on which the annotations were performed (usually the full
        resolution image). This can be the full image datasets, but the
        PixelData attributes are not required.
    graphic_type: Union[highdicom.ann.GraphicTypeValues, str], optional
        Graphic type to use to store all nuclei. Note that all but 'POLYGON'
        result in simplification and loss of information in the annotations.
    annotation_coordinate_type: Union[hd.ann.AnnotationCoordinateTypeValues, str], optional
        Store coordinates in the Bulk Microscopy Bulk Simple Annotations in the
        (3D) frame of reference (SCOORD3D), or the (2D) total pixel matrix
        (SCOORD, default).

    Returns
    -------
    polygons: list[shapely.geometry.polygon.Polygon]
        List of polygons. Note that this is always the list of full original,
        2D polygons regardless of the requested graphic type and annotation
        coordinate type.
    graphic_data: list[np.ndarray]
        List of graphic data as numpy arrays in the format required for the
        MicroscopyBulkSimpleAnnotations object. These are correctly formatted
        for the requested graphic type and annotation coordinate type.
    identifier: list[int]
        Identifiers for each annotation taken as is from input. Does not depend
        on requested graphic type or coordinate type.
    """  
    graphic_type = hd.ann.GraphicTypeValues[graphic_type]
    annotation_coordinate_type = hd.ann.AnnotationCoordinateTypeValues[
        annotation_coordinate_type
    ]

    transformer = hd.spatial.ImageToReferenceTransformer.for_image(
        source_image_metadata,
        for_total_pixel_matrix=True,
    )

    if graphic_type == hd.ann.GraphicTypeValues.POLYLINE:
        raise ValueError('Graphic type 'POLYLINE' not supported.')

    graphic_data = []
    polygons = []
    areas = []

    for df_row in annotations.iterrows():
        polygon, graphic_item, area = process_df_row(
            df_row,
            transformer,
            area_per_pixel_um2,
            graphic_type,
            annotation_coordinate_type,
        )

        if polygon is None:
            continue

        graphic_data.append(graphic_item)
        polygons.append(polygon)

        areas.append(area)

    logging.info(f'Parsed {len(graphic_data)} annotations.')

    return polygons, graphic_data, areas


def create_bulk_annotations(
    source_image_metadata: Dataset,
    graphic_data: list[np.ndarray],
    areas: list[float],
    graphic_type: Union[
        hd.ann.GraphicTypeValues,
        str
    ] = hd.ann.GraphicTypeValues.POLYGON,
    annotation_coordinate_type: Union[
        hd.ann.AnnotationCoordinateTypeValues,
        str
    ] = hd.ann.AnnotationCoordinateTypeValues.SCOORD,
) -> hd.ann.MicroscopyBulkSimpleAnnotations:
    """

    Parameters
    ----------
    source_image_metadata: pydicom.Dataset
        Metadata of the image from which annotations were derived.
    graphic_data: list[np.ndarray]
        Pre-computed graphic data for the graphic type and annotation
        coordinate type.
    areas: list[float]
        Area measurement in square micrometers for each annotation.
    graphic_type: Union[highdicom.ann.GraphicTypeValues, str], optional
        Graphic type to use to store all nuclei. Note that all but 'POLYGON'
        result in simplification and loss of information in the annotations.
    annotation_coordinate_type: Union[hd.ann.AnnotationCoordinateTypeValues, str], optional
        Store coordinates in the Bulk Microscopy Bulk Simple Annotations in the
        (3D) frame of reference (SCOORD3D), or the (2D) total pixel matrix
        (SCOORD, default).

    Returns
    -------
    annotation: hd.ann.MicroscopyBulkSimpleAnnotations:
        DICOM bulk microscopy annotation encoding the original annotations in
        vector format.

    """
    graphic_type = hd.ann.GraphicTypeValues[graphic_type]
    annotation_coordinate_type = hd.ann.AnnotationCoordinateTypeValues[
        annotation_coordinate_type
    ]

    # TODO: for loop for the different possible labels
    group = hd.ann.AnnotationGroup(
        number=1,
        uid=hd.UID(),
        label=metadata_config.label,
        annotated_property_category=metadata_config.finding_category,
        annotated_property_type=metadata_config.finding_type,
        graphic_type=graphic_type,
        graphic_data=graphic_data,
        algorithm_type=hd.ann.AnnotationGroupGenerationTypeValues.AUTOMATIC,
        algorithm_identification=metadata_config.algorithm_identification,
        measurements=[
            hd.ann.Measurements(
                name=codes.SCT.Area,
                unit=codes.UCUM.SquareMicrometer,
                values=np.array(areas),
            )
        ],
    )
    annotations = hd.ann.MicroscopyBulkSimpleAnnotations(
        source_images=[source_image_metadata],
        annotation_coordinate_type=annotation_coordinate_type,
        annotation_groups=[group],
        series_instance_uid=hd.UID(),
        series_number=204,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer=metadata_config.manufacturer,
        manufacturer_model_name=metadata_config.manufacturer_model_name,
        software_versions=metadata_config.software_versions,
        device_serial_number=metadata_config.device_serial_number,
    )
    annotations.add(metadata_config.other_trials_seq_element)

    return annotations
