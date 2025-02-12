"""Utilities for converting annotations. Clear of cloud-specific things."""
import logging
import numpy as np
import highdicom as hd
from pydicom import Dataset
from pydicom.sr.codedict import codes
from shapely.geometry import box
from typing import List, Tuple

import metadata_config
from data_utils import CellAnnotation 


def process_annotation(
    ann: CellAnnotation,
    transformer: hd.spatial.ImageToReferenceTransformer,
    graphic_type: hd.ann.GraphicTypeValues,
    annotation_coordinate_type: hd.ann.AnnotationCoordinateTypeValues
    ) -> Tuple[np.ndarray, int, str]:
    """
    Process a single annotation to be in the format required for DICOM ANN files.

    Parameters
    ----------
    ann: CellAnnotation
        Single annotation. 
    transformer: hd.spatial.ImageToReferenceTransformer
        Transformer object to map image coordinates to reference coordinates
        for the image.
    graphic_type: hd.ann.GraphicTypeValues, optional 
        Graphic type to use to store all nuclei. Allowed options are 'POLYGON' (default)
        or 'POINT'. 
    annotation_coordinate_type: hd.ann.AnnotationCoordinateTypeValues, optional
        Store coordinates in the Bulk Microscopy Bulk Simple Annotations in the
        (3D) frame of reference (SCOORD3D), or the (2D) total pixel matrix
        (SCOORD, default).

    Returns
    -------
    graphic_data: np.ndarray
        Numpy array of float32 coordinates to include in the Bulk Microscopy
        Simple Annotations.
    identifier: list[int]
        Identifier taken as is from input. 
    labels: list[str]
        Label taken as is from input.
    """      
    bounding_box = box(*ann.bounding_box)

    # Remove the final point (require not to be closed but Polygon adds this)
    coords = np.array(bounding_box.exterior.coords)[:-1, :]

    # Simplify the coordinates as required
    if graphic_type == hd.ann.GraphicTypeValues.POLYGON:
        graphic_data = coords
    elif graphic_type == hd.ann.GraphicTypeValues.POINT:
        x, y = bounding_box.centroid.xy
        graphic_data = np.array([[x[0], y[0]]])
    else:
        raise ValueError(
            f'Graphic type "{graphic_type.value}" not supported.'
        )

    use_3d = (
        annotation_coordinate_type ==
        hd.ann.AnnotationCoordinateTypeValues.SCOORD3D
    )
    if use_3d:
        graphic_data = transformer(graphic_data)

    return graphic_data.astype(np.float32), ann.identifier, ann.label


def get_graphic_data(
    annotations: List[CellAnnotation],
    source_image_metadata: Dataset,
    graphic_type: str = 'POLYGON',
    annotation_coordinate_type: str = 'SCOORD'
    ) -> Tuple[list[np.ndarray], list[int], list[str]]:
    """
    Parse annotations to construct graphic data.

    Parameters
    ----------
    annotations: List[CellAnnotations]
        List of annotations for the slide. 
    source_image_metadata: pydicom.Dataset
        Pydicom datasets containing the metadata of the reference image (already
        converted to DICOM format). This can be the full image datasets, but the
        PixelData attributes are not required.
    graphic_type: str, optional 
        Graphic type to use to store all nuclei. Allowed options are 'POLYGON' (default)
        or 'POINT'.
    annotation_coordinate_type: str, optional
        Store coordinates in the Bulk Microscopy Bulk Simple Annotations in the
        (3D) frame of reference (SCOORD3D), or the (2D) total pixel matrix
        (SCOORD, default).

    Returns
    -------
    graphic_data: list[np.ndarray]
        List of graphic data as numpy arrays in the format required for the
        MicroscopyBulkSimpleAnnotations object. These are correctly formatted
        for the requested graphic type and annotation coordinate type.
    identifiers: list[int]
        Identifier for each annotation taken as is from input. 
    labels: list[str]
        Label for each annotation taken as is from input. 
    """  
    graphic_type = hd.ann.GraphicTypeValues[graphic_type]
    annotation_coordinate_type = hd.ann.AnnotationCoordinateTypeValues[
        annotation_coordinate_type
    ]

    transformer = hd.spatial.ImageToReferenceTransformer.for_image(
        source_image_metadata,
        for_total_pixel_matrix=True,
    )

    graphic_data = []
    identifiers = []
    labels = []

    for ann in annotations:
        graphic_item, identifier, label = process_annotation(
            ann,
            transformer,
            graphic_type,
            annotation_coordinate_type,
        )

        graphic_data.append(graphic_item)
        identifiers.append(identifier)
        labels.append(label)

    logging.info(f'Parsed {len(graphic_data)} annotations.')

    return graphic_data, identifiers, labels


def create_bulk_annotations(
    source_image_metadata: Dataset,
    graphic_data: list[np.ndarray],
    identifiers: list[int],
    labels: list[str],
    graphic_type: str = 'POLYGON',
    annotation_coordinate_type: str = 'SCOORD'
    ) -> hd.ann.MicroscopyBulkSimpleAnnotations:
    """
    Create DICOM Microscopy Bulk Simple Annotation objects. 

    Parameters
    ----------
    source_image_metadata: pydicom.Dataset
        Metadata of the image from which annotations were derived.
    graphic_data: list[np.ndarray]
        Pre-computed graphic data for the graphic type and annotation
        coordinate type.
    identifiers: list[int]
        Identifier for each annotation taken as is from input. 
    labels: list[str]
        Label for each annotation taken as is from input. 
    graphic_type: str, optional 
        Graphic type to use to store all nuclei. Allowed options are 'POLYGON' (default)
        or 'POINT'.
    annotation_coordinate_type: hd.ann.AnnotationCoordinateTypeValues, optional
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

    groups = []
    group_number = 1
    for label_idx, label in enumerate(sorted(metadata_config.labels_dict['label'])): 
        indices = np.where(np.array(labels) == label)[0].tolist()
        if len(indices) > 0: 
            group = hd.ann.AnnotationGroup(
                number=group_number,
                uid=hd.UID(),
                label=label,
                annotated_property_category=metadata_config.labels_dict['finding_category'][label_idx],
                annotated_property_type=metadata_config.labels_dict['finding_type'][label_idx],
                graphic_type=graphic_type,
                graphic_data=[graphic_data[i] for i in indices],
                algorithm_type=hd.ann.AnnotationGroupGenerationTypeValues.AUTOMATIC,
                algorithm_identification=metadata_config.algorithm_identification,
                measurements=[
                    hd.ann.Measurements(
                        name=codes.SCT.Area,
                        unit=codes.UCUM.SquareMicrometer,
                        values=np.array([identifiers[i] for i in indices]),
                    )
                ],
            )
            groups.append(group)
            group_number += 1

    annotations = hd.ann.MicroscopyBulkSimpleAnnotations(
        source_images=[source_image_metadata],
        annotation_coordinate_type=annotation_coordinate_type,
        annotation_groups=groups,
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
