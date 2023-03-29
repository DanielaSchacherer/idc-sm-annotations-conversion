"""Utilities for converting annotations. Keep clear of cloud-specific things."""
import logging
from os import PathLike
from socket import gethostname
from typing import Iterable, Optional, Sequence, Tuple, Union

import highdicom as hd
import mahotas as mh
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
from pydicom import Dataset
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code
from pydicom.uid import JPEGLSLossless, ExplicitVRLittleEndian
from shapely.geometry.polygon import Polygon


def compute_tile_positions(source_image: Dataset) -> np.ndarray:
    """Compute the positions of each frame within the total pixel matrix.

    Parameters
    ----------
    source_image : pydicom.Dataset
        Metadata of a tiled image

    Returns
    -------
    numpy.ndarray
        (Column, Row) position of each frame in the total pixel matrix in pixel
        unit. The top left frame is located at (1, 1).

    """
    if hasattr(source_image, "PerFrameFunctionalGroupsSequence"):
        plane_positions = [
            hd.PlanePositionSequence.from_sequence(
                frame_item.PlanePositionSlideSequence
            )
            for frame_item in source_image.PerFrameFunctionalGroupsSequence
        ]
    else:
        plane_positions = hd.utils.compute_plane_position_slide_per_frame(
            source_image
        )
    return np.array(
        [
            (
                int(seq[0].ColumnPositionInTotalImagePixelMatrix),
                int(seq[0].RowPositionInTotalImagePixelMatrix),
            )
            for seq in plane_positions
        ],
        dtype=int,
    )


def disassemble_total_pixel_matrix(
    total_pixel_matrix: np.ndarray,
    tile_positions: Sequence[Tuple[int, int]],
    rows: int,
    columns: int,
) -> np.ndarray:
    """Disassemble a total pixel matrix into individual tiles.

    Parameters
    ----------
    total_pixel_matrix: numpy.ndarray
        Total pixel matrix
    tile_positions: Sequence[Tuple[int, int]]
        Column, Row position of each tile relative to the slide
    rows: int
        Number of rows per tile
    columns: int
        Number of columns per tile

    Returns
    -------
    numpy.ndarray
        Stacked image tiles

    """
    tiles = []
    if total_pixel_matrix.ndim == 3:
        tile_shape = (rows, columns, total_pixel_matrix.shape[-1])
    elif total_pixel_matrix.ndim == 2:
        tile_shape = (rows, columns)
    else:
        raise ValueError(
            "Total pixel matrix has unexpected number of dimensions."
        )
    for row_offset, column_offset in tile_positions:
        tile = np.zeros(tile_shape, dtype=total_pixel_matrix.dtype)
        pixel_array = total_pixel_matrix[
            row_offset: (row_offset + rows),
            column_offset: (column_offset + columns),
            ...,
        ]
        tile[
            0:pixel_array.shape[0], 0:pixel_array.shape[1], ...
        ] = pixel_array
        tiles.append(tile)
    return np.stack(tiles)


def convert_annotations(
    annotation_csvs: Iterable[Union[str, PathLike]],
    source_image_metadata: Dataset,
    *,
    include_segmentation: bool = False,
    store_boundary: bool = True,
    annotation_coordinate_type: Union[
        hd.ann.AnnotationCoordinateTypeValues,
        str
    ] = hd.ann.AnnotationCoordinateTypeValues.SCOORD,
    segmentation_type: Union[
        hd.seg.SegmentationTypeValues,
        str
    ] = hd.seg.SegmentationTypeValues.BINARY,
    debug: bool = False,
) -> Tuple[
    hd.ann.MicroscopyBulkSimpleAnnotations,
    Optional[hd.seg.Segmentation]
]:
    """Convert an annotation into DICOM format.

    Specifically, a Bulk Microscopy Bulk Simple Annotation object (vector
    graphics) is created and a Segmentation Image (raster) is optionally
    created.

    Parameters
    ----------
    annotation_csvs: Iterable[Union[str, os.PathLike]]
        Iterable over pathlike objects, each representing the path to a
        CSV-format file containing an annotation for this image.
    source_image_metadata: pydicom.Dataset
        Pydicom dataset containing the metadata of the image (already converted
        to DICOM format). Note that this should be the metadata of the image
        at full resolution. This can be the full image dataset, but the
        PixelData attribute is not required.
    include_segmentation: bool, optional
        Include the segmentation output.
    store_boundary: bool, optional
        Store the full nucleus boundary polygon in the Bulk Microscopy Bulk
        Simple Annotations. If False, just the centroid is stored as a single
        point.
    annotation_coordinate_type: Union[hd.ann.AnnotationCoordinateTypeValues, str], optional
        Store coordinates in the Bulk Microscopy Bulk Simple Annotations in the
        (3D) frame of reference (SCOORD3D), or the (2D) total pixel matrix
        (SCOORD, default).
    segmentation_type: Union[hd.seg.SegmentationTypeValues, str], optional
        Segmentation type (BINARY or FRACTIONAL) for the Segmentation Image
        (if any).
    debug: bool, optional
        Show visualizations using matplotlib for debugging.

    Returns
    -------
    annotation: hd.ann.MicroscopyBulkSimpleAnnotations:
        DICOM bulk microscopy annotation encoding the original annotations in
        vector format.
    segmentation: Optional[hd.seg.Segmentation]:
        DICOM segmentation image encoding the original annotations in raster
        format, if requested. None otherwise.

    """  # noqa: E501
    segmentation_type = hd.seg.SegmentationTypeValues[segmentation_type]
    annotation_coordinate_type = hd.ann.AnnotationCoordinateTypeValues[
        annotation_coordinate_type
    ]

    image_orientation = source_image_metadata.ImageOrientationSlide
    origin = source_image_metadata.TotalPixelMatrixOriginSequence[0]
    image_position = (
        float(origin.XOffsetInSlideCoordinateSystem),
        float(origin.YOffsetInSlideCoordinateSystem),
        0.0,
    )
    pixel_spacing = (
        source_image_metadata.SharedFunctionalGroupsSequence[0]
        .PixelMeasuresSequence[0]
        .PixelSpacing
    )
    transformer = hd.spatial.ImageToReferenceTransformer(
        image_orientation=image_orientation,
        image_position=image_position,
        pixel_spacing=pixel_spacing,
    )

    graphic_type = (
        hd.ann.GraphicTypeValues.POLYGON
        if store_boundary
        else hd.ann.GraphicTypeValues.POINT
    )
    graphic_data = []
    measurements = {
        (codes.SCT.Area, codes.UCUM.SquareMicrometer): [],
    }

    if include_segmentation:
        segmentation_mask = np.zeros(
            (
                source_image_metadata.TotalPixelMatrixRows,
                source_image_metadata.TotalPixelMatrixColumns,
            ),
            dtype=bool
        )
        logging.info(
            f"Total Pixel Matrix {segmentation_mask.shape} "
            f"{segmentation_mask.nbytes:.3g} bytes."
        )

    for f in annotation_csvs:
        df = pd.read_csv(f)
        if debug:
            fig, axes = plt.subplots(1, 2)
            min_offsets = np.zeros((1, 2), dtype=np.float32)
            min_offsets[0, 0] = source_image_metadata.TotalPixelMatrixRows
            min_offsets[0, 1] = source_image_metadata.TotalPixelMatrixColumns
            max_offsets = np.zeros((1, 2), dtype=np.float32)

        for i, (index, values) in enumerate(df.iterrows()):
            points = np.array(
                values.Polygon[1:-1].split(':'),
                dtype=np.float32
            )
            n = len(points) // 2
            coordinates_image = points.reshape(n, 2)
            if coordinates_image.shape[0] < 3:
                continue
            polygon_image = Polygon(coordinates_image)
            c, r = polygon_image.exterior.xy

            if debug:
                axes[1].plot(c, r, color='#027ea3')
                min_offsets = np.minimum(
                    min_offsets,
                    coordinates_image.min(axis=0)
                )
                max_offsets = np.maximum(
                    max_offsets,
                    coordinates_image.max(axis=0)
                )

            if include_segmentation:
                contour_image = np.stack([r, c]).T.astype(np.int32)
                mh.polygon.fill_polygon(contour_image, segmentation_mask)

            # fig, axes = plt.subplots(1, 2)
            # axes[1].plot(c, r, color='#027ea3')
            # axes[1].invert_yaxis()
            # contour_obj = contour_image - contour_image.min(axis=0)
            # segmentation_obj_mask = np.zeros(contour_obj.max(axis=0), np.bool)
            # mh.polygon.fill_polygon(contour_obj, segmentation_obj_mask)
            # axes[0].imshow(segmentation_obj_mask)
            # name = f'{f.stem} - {i}'
            # fig.suptitle(name)
            # fig.savefig(f'/tmp/{name}.pdf')

            coordinates_ref = transformer(coordinates_image)
            polygon_ref = Polygon(coordinates_ref)

            if store_boundary:
                # Store the full polygon in graphic data
                if (
                    annotation_coordinate_type ==
                    hd.ann.AnnotationCoordinateTypeValues.SCOORD3D
                ):
                    coords = np.array(polygon_ref.exterior.coords)
                else:
                    # 2D total pixel matrix coordinates
                    coords = np.array(polygon_image.exterior.coords)

                # Remove the final point (polygon should not be closed)
                coords = coords[:-1, :]
                graphic_data.append(coords)
            else:
                # Store the centroid of the polygon only, as a single point
                if (
                    annotation_coordinate_type ==
                    hd.ann.AnnotationCoordinateTypeValues.SCOORD3D
                ):
                    x, y = polygon_ref.centroid.xy
                    centroid = np.array([[x[0], y[0], 0.]])
                else:
                    # 2D total pixel matrix coordinates
                    x, y = polygon_image.centroid.xy
                    centroid = np.array([[x[0], y[0]]])
                graphic_data.append(centroid)

            area = float(polygon_ref.area)
            measurements[(codes.SCT.Area, codes.UCUM.SquareMicrometer)].append(
                area
            )

        if debug and include_segmentation:
            segmentation_frame_mask = segmentation_mask[
                int(min_offsets[0, 0]):int(max_offsets[0, 0]),
                int(min_offsets[0, 1]):int(max_offsets[0, 1])
            ]
            axes[0].imshow(segmentation_frame_mask)
            plt.show()

    logging.info(f"Parsed {len(graphic_data)} annotations.")

    name = 'Pan-Cancer-Nuclei-Seg'
    algorithm_identification = hd.AlgorithmIdentificationSequence(
        name=name,
        version='1.0',
        family=codes.cid7162.ArtificialIntelligence,
    )

    finding_category = Code("91723000", "SCT", "Anatomical Stucture")
    finding_type = Code("84640000", "SCT", "Nucleus")

    logging.info("Creating annotation.")
    group = hd.ann.AnnotationGroup(
        number=1,
        uid=hd.UID(),
        label='nuclei',
        annotated_property_category=finding_category,
        annotated_property_type=finding_type,
        graphic_type=graphic_type,
        graphic_data=graphic_data,
        algorithm_type=hd.ann.AnnotationGroupGenerationTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        measurements=[
            hd.ann.Measurements(
                name=name,
                unit=unit,
                values=np.array(values),
            )
            for (name, unit), values in measurements.items()
        ]
    )
    annotations = hd.ann.MicroscopyBulkSimpleAnnotations(
        source_images=[source_image_metadata],
        annotation_coordinate_type=annotation_coordinate_type,
        annotation_groups=[group],
        series_instance_uid=hd.UID(),
        series_number=204,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer='MGH Computational Pathology',
        manufacturer_model_name="tumor-classification",
        software_versions='1.0',
        device_serial_number=gethostname()
    )

    if include_segmentation:
        logging.info("Creating segmentation.")
        tile_positions = compute_tile_positions(source_image_metadata)
        segmentation_tiles = disassemble_total_pixel_matrix(
            total_pixel_matrix=segmentation_mask,
            tile_positions=tile_positions,
            rows=source_image_metadata.Rows,
            columns=source_image_metadata.Columns
        )

        del segmentation_mask  # help free up some memory

        segment_description = hd.seg.SegmentDescription(
            segment_number=1,
            segment_label='Nuclei',
            segmented_property_category=finding_category,
            segmented_property_type=finding_type,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=algorithm_identification
        )

        # Compression method depends on what is possible given the chosen
        # segmentation type
        transfer_syntax_uid = {
            hd.seg.SegmentationTypeValues.BINARY: ExplicitVRLittleEndian,
            hd.seg.SegmentationTypeValues.FRACTIONAL: JPEGLSLossless,
        }[segmentation_type]

        segmentation = hd.seg.Segmentation(
            source_images=[source_image_metadata],
            pixel_array=segmentation_tiles,
            segmentation_type=segmentation_type,
            segment_descriptions=[segment_description],
            series_instance_uid=hd.UID(),
            series_number=20,
            sop_instance_uid=hd.UID(),
            instance_number=1,
            manufacturer='MGH Computational Pathology',
            manufacturer_model_name="tumor-classification",
            software_versions='1.0',
            device_serial_number=gethostname(),
            transfer_syntax_uid=transfer_syntax_uid,
        )
    else:
        segmentation = None

    return annotations, segmentation
