"""Entrypoint for annotation conversion."""
import os
import datetime
import pydicom 
import logging
import pandas as pd
from pathlib import Path
from time import time
from typing import Any, Dict 

from data_utils import CellAnnotation, preprocess_annotation_csvs, filter_cell_annotations
from convert import (
    get_graphic_data,
    create_bulk_annotations,
)


def get_source_image_metadata(slide_dir: Path) -> Dict[str, Any]: 
    """ 
    Function that finds the source image file (base level) and extracts relevant metadata. 
    
    Parameters
    ----------
    slide_dir: Path
        Folder containing image data for the respective slide

    Returns
    -------
    data: dict[str, Any]
        Output data packed into a dict. This will contain 
        - slide_id: str 
        - source_image: pydicom.Dataset 
            pydicom-read metadata excluding pixel data
    """

    def find_base_level(dcm_dir: Path) -> Path:
        """ Find base level (i.e. largest file) in folder. """ 
        base_level = None
        largest_size = 0
        for level in [p.resolve() for p in dcm_dir.iterdir()]: 
            level_size = os.path.getsize(level)
            if level_size > largest_size: 
                largest_size = level_size
                base_level = level
        return base_level

    base_level = find_base_level(slide_dir)
    ds = pydicom.dcmread(base_level, stop_before_pixels=True)
    
    data = dict( 
        slide_id = slide_dir.stem, 
        source_image = ds
    )
    return data 


def parse_annotations(data: Dict[str, Any], annotations: pd.DataFrame) -> Dict[str, Any]: 
    """ 
    Parses annotations from pd.DataFrame into a list of CellAnnotations. 

    Parameters
    ----------
    data: dict[str, Any]
        Input data packed into a dict, including at least:
        - slide_id: str
            Slide ID for the case.
        - source_image: pydicom.Dataset
            Base level source image for this case.

    Returns
    -------
    data: dict[str, Any]
        Output data packed into a dict. This will contain the same keys as
        the input dictionary, plus the following additional keys:
        - ann: list[CellAnnotation]
            List of annotations. 
    """

    ann = []
    for _, row in annotations.iterrows(): 
        x_min, y_min = row['x_in_slide'], row['y_in_slide']
        x_max, y_max = x_min + row['cell_width'], y_min + row['cell_height']
        ann.append(CellAnnotation(
            identifier=row['cell_id'], 
            bounding_box=(x_min, y_min, x_max, y_max), 
            label=row['all_original_annotations'].split(',')[0]
        ))

    data['ann'] = ann
    return data


def parse_annotations_to_graphic_data(
    data: Dict[str, Any],
    graphic_type: str, 
    annotation_coordinate_type: str,
    output_dir: Path
    ) -> Dict[str, Any]:

    """
    Parse annotations to highdicom graphic data.

    Parameters
    ----------
    data: dict[str, Any]
        Input data packed into a dict, including at least:
        - slide_id: str
            Slide ID for the case.
        - source_image: pydicom.Dataset
            Base level source image for this case.
        - ann: list[CellAnnotations]
            List of cell annotations. 
    graphic_type: hd.ann.GraphicTypeValues
        Graphic type to use to store all nuclei. Allowed options are 'POLYGON' (default)
        or 'POINT' (will be bounding box centroid). 
    annotation_coordinate_type: hd.ann.AnnotationCoordinateTypeValues
        Store coordinates in the Bulk Microscopy Bulk Simple Annotations in
        the (3D) frame of reference (SCOORD3D), or the (2D) total pixel
        matrix (SCOORD, default).
    output_dir: pathlib.Path
        A local output directory to store error logs.

    Returns
    -------
    data: dict[str, Any]
        Output data packed into a dict. This will contain the same keys as
        the input dictionary, plus the following additional keys:

        - graphic_data: list[np.ndarray]
            List of graphic data as numpy arrays in the format required for
            the MicroscopyBulkSimpleAnnotations object. These are correctly
            formatted for the requested graphic type and annotation
            coordinate type.
        - identifiers: list[int]
            Identifier for each of the bounding boxes. The identifier is a consecutive number 
            going over the whole dataset (not only a single slide).

    """

    errors = []
    slide_id = data['slide_id']

    start_time = time()
    logging.info(f'Parsing annotations for slide: {slide_id}')
    try:
        graphic_data, identifiers, labels = get_graphic_data(
            annotations=data['ann'],
            source_image_metadata=data['source_image'],
            graphic_type=graphic_type,
            annotation_coordinate_type=annotation_coordinate_type,
        )
    except Exception as e:
        logging.error(f'Error {str(e)}')
        errors.append(
            {
                'slide_id': slide_id,
                'error_message': str(e),
                'datetime': str(datetime.datetime.now()),
            }
        )
        errors_df = pd.DataFrame(errors)
        errors_df.to_csv(output_dir / 'conversion_error_log.txt')
        return None

    stop_time = time()
    duration = stop_time - start_time
    logging.info(
        f'Processed annotations for slide {slide_id} in {duration:.2f}s'
    )

    del data['ann'] # save memory 
    data['graphic_data'] = graphic_data
    data['identifiers'] = identifiers
    data['labels'] = labels
    return data


def create_dcm_annotations(
    data: Dict[str, Any],
    graphic_type: str,
    annotation_coordinate_type: str,
    output_dir: Path
    ) -> Dict[str, Any]:

    """
    Creates bulk annotations DICOM objects.

    Parameters
    ----------
    data: dict[str, Any]
        Input data packed into a dict, including at least:
        - slide_id: str
            Slide ID for the case.
        - graphic_data: list[np.ndarray]
            List of graphic data as numpy arrays in the format required for
            the MicroscopyBulkSimpleAnnotations object. These are correctly
            formatted for the requested graphic type and annotation
            coordinate type.
        - source_image: pydicom.Dataset
            Base level source image for this case.
        - identifiers: list[int]
            Identifier for each of the bounding boxes. The identifier is a consecutive number 
            going over the whole dataset (not only a single slide).
        - labels: list[str]
            Label for each bounding box. 
    graphic_type: str
        Graphic type to use to store all nuclei. Allowed options are 'POLYGON' (default)
        or 'POINT'.
    annotation_coordinate_type: str
        Store coordinates in the Bulk Microscopy Bulk Simple Annotations in
        the (3D) frame of reference (SCOORD3D), or the (2D) total pixel
        matrix (SCOORD, default).
    output_dir: pathlib.Path
        A local output directory to store error logs.

    Returns
    -------
    data: dict[str, Any]
        Output data packed into a dict. This will contain the same keys as
        the input dictionary, plus the following additional keys:

        - ann_dcm: hd.ann.MicroscopyBulkSimpleAnnotations:
            DICOM bulk microscopy annotation encoding the original
            annotations in vector format.

    """

    errors = []
    slide_id = data['slide_id']
    start_time = time()
    logging.info(f'Creating annotation for slide: {slide_id}')

    try:
        ann_dcm = create_bulk_annotations(
            source_image_metadata=data['source_image'],
            graphic_data=data['graphic_data'],
            identifiers=data['identifiers'],
            labels=data['labels'],
            graphic_type=graphic_type,
            annotation_coordinate_type=annotation_coordinate_type
        )
    except Exception as e:
        logging.error(f"Error {str(e)}")
        errors.append(
            {
                'slide_id': slide_id,
                'error_message': str(e),
                'datetime': str(datetime.datetime.now()),
            }
        )
        errors_df = pd.DataFrame(errors)
        errors_df.to_csv(output_dir / 'annotation_creator_error_log.txt')
        return None

    stop_time = time()
    duration = stop_time - start_time
    logging.info(
        f'Created annotation for for slide {slide_id} in {duration:.2f}s'
    )

    
    del data['graphic_data'] # Save some memory
    del data['identifiers'] # Save some memory
    del data['labels'] # Save some memory
    data['ann_dcm'] = ann_dcm
    return data


def save_annotations(
    data: dict[str, Any],
    output_dir: Path
    ) -> None: 

    """
    Store files.

    Parameters
    ----------
    data: dict[str, Any]
        Input data packed into a dictionary, containing at least:

        - slide_id: str
            Slide ID for the case.
        - ann_dcm: highdicom.ann.MicroscopyBulkSimpleAnnotations
            Bulk Annotation DICOM object.

    output_dir: pathlib.Path
        A local output directory to store the downloaded files and error logs. 
    """
    errors = []
    slide_id = data['slide_id']
    
    image_start_time = time()
    logging.info(f'Saving annotations for slide {slide_id}')
    
    slide_ann_dir = output_dir / slide_id
    slide_ann_dir.mkdir(exist_ok=True)

    try:
        ann_path = f'{slide_ann_dir}/{slide_id}_ann.dcm'
        logging.info(f'Writing annotation to {str(ann_path)}.')
        data['ann_dcm'].save_as(ann_path)

        image_stop_time = time()
        time_for_image = image_stop_time - image_start_time
        logging.info(
            f'Saved annotations for slide {slide_id} in {time_for_image:.2f}s'
        )
        
    except Exception as e:
        logging.error(f"Error {str(e)}")
        errors.append(
            {
                'slide_id': slide_id,
                'error_message': str(e),
                'datetime': str(datetime.datetime.now()),
            }
        )
        errors_df = pd.DataFrame(errors)
        errors_df.to_csv(output_dir / 'save_error_log.txt')
        return None


def run(
    csv_cells: Path, 
    csv_rois: Path, 
    source_image_root_dir: Path,
    output_dir: Path,
    graphic_type: str = 'POLYGON',
    annotation_coordinate_type: str = 'SCOORD',
) -> None: 
    
    logging.basicConfig(level=logging.INFO)

    # Suppress highdicom logging (very talkative)
    logging.getLogger('highdicom.base').setLevel(logging.WARNING)
    logging.getLogger('highdicom.seg.sop').setLevel(logging.WARNING)
    
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)
    
    csv_cells = preprocess_annotation_csvs(csv_cells, csv_rois)

    for slide_id in os.listdir(source_image_root_dir): 
        slide_cells = filter_cell_annotations(csv_cells, slide_id)
        if len(slide_cells) > 0: 
            data = get_source_image_metadata(source_image_root_dir/slide_id)
            data = parse_annotations(data, slide_cells)
            data = parse_annotations_to_graphic_data(data, graphic_type, annotation_coordinate_type, output_dir)
            data = create_dcm_annotations(data, graphic_type, annotation_coordinate_type, output_dir)
            save_annotations(data, output_dir)


if __name__ == "__main__":
    data_dir = Path('/home/dschacherer/bmdeep_conversion/data/bmdeep_DICOM_converted')
    slide_id = 'F7177163C833DFF4B38FC8D2872F1EC6_1_bm'
    cell_csv = Path('/home/dschacherer/bmdeep_conversion/data/cells.csv')
    roi_csv = Path('/home/dschacherer/bmdeep_conversion/data/rois.csv')
    run(cell_csv, roi_csv, data_dir, data_dir)

