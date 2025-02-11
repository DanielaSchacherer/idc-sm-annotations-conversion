"""Entrypoint for Pan Cancer Nuclei Segmentation annotation conversion."""
import os
import datetime
import pydicom 
import logging
import pandas as pd
from pathlib import Path
from time import time
from typing import Any, Dict

from idc_annotation_conversion.pan_cancer_nuclei_seg.convert import (
    get_graphic_data,
    create_bulk_annotations,
)

ANNOTATION_PREFIX = 'cnn-nuclear-segmentations-2019/data-files/'

def preprocess_annotation_csvs(cells_csv: Path, roi_csv: Path) -> pd.DataFrame: 
    """ 
    Function to massage the annotation data to fit the required format for the conversion process.
    """

    cells = pd.read_csv(cells_csv)
    rois = pd.read_csv(roi_csv)
    return pd.merge(cells, rois[['id', 'slide_id']], 
                    left_on='rocellboxing_id', 
                    right_on = 'id', 
                    how='left').drop('id', axis=1)


def filter_cell_annotations(annotations: pd.DataFrame, slide_id: str) -> pd.DataFrame: 
    """ 
    Function to filter for annotations of a single slide.
    """
    return annotations[annotations['slide_id'] == slide_id] 


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


class AnnotationParser:

    """Class that parses CSV annotations to graphic data."""

    def __init__(
        self,
        annotation_coordinate_type: str,
        graphic_type: str, 
        output_dir: Path
    ):
        """

        Parameters
        ----------
        annotation_coordinate_type: str
            Store coordinates in the Bulk Microscopy Bulk Simple Annotations in
            the (3D) frame of reference (SCOORD3D), or the (2D) total pixel
            matrix (SCOORD, default).
        graphic_type: str
            Graphic type to use to store all nuclei. Note that all but
            'POLYGON' result in simplification and loss of information in the
            annotations.
        output_dir: pathlib.Path
            A local output directory to store error logs.

        """
        self._annotation_coordinate_type = annotation_coordinate_type
        self._graphic_type = graphic_type
        self._output_dir = output_dir
        self._errors = []

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Parse CSV to graphic data.

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

            - polygons: list[shapely.geometry.polygon.Polygon]
                List of polygons. Note that this is always the list of full
                original, 2D polygons regardless of the requested graphic type
                and annotation coordinate type.
            - graphic_data: list[np.ndarray]
                List of graphic data as numpy arrays in the format required for
                the MicroscopyBulkSimpleAnnotations object. These are correctly
                formatted for the requested graphic type and annotation
                coordinate type.
            - identifiers: list[int]
                Identifier for each of the polygons. The identifier is a consecutive number 
                going over the whole dataset (not only a single slide). 
        """

        start_time = time()
        slide_id = data['slide_id']
        logging.info(f'Parsing annotations for slide: {slide_id}')
        try:
            polygons, graphic_data, identifiers = get_graphic_data(
                annotations='tbd',
                source_image_metadata=data['source_image'],
                graphic_type=self._graphic_type,
                annotation_coordinate_type=self._annotation_coordinate_type,
                workers=self._workers,
            )
        except Exception as e:
            logging.error(f'Error {str(e)}')
            self._errors.append(
                {
                    'slide_id': data['slide_id'],
                    'error_message': str(e),
                    'datetime': str(datetime.datetime.now()),
                }
            )
            errors_df = pd.DataFrame(self._errors)
            errors_df.to_csv(self._output_dir / 'conversion_error_log.csv')
            return None

        stop_time = time()
        duration = stop_time - start_time
        logging.info(
            f'Processed annotations for slide {slide_id} in {duration:.2f}s'
        )

        data['polygons'] = polygons
        data['graphic_data'] = graphic_data
        data['identifiers'] = identifiers
        return data


class AnnotationCreator:

    """Class that creates bulk annotations DICOM objects."""

    def __init__(
        self,
        graphic_type: str,
        annotation_coordinate_type: str,
        output_dir: Path
    ):
        """

        Parameters
        ----------
        graphic_type: str
            Graphic type to use to store all nuclei. Note that all but
            'POLYGON' result in simplification and loss of information in the
            annotations.
        annotation_coordinate_type: str
            Store coordinates in the Bulk Microscopy Bulk Simple Annotations in
            the (3D) frame of reference (SCOORD3D), or the (2D) total pixel
            matrix (SCOORD, default).
        output_dir: pathlib.Path
            A local output directory to store error logs.

        """
        self._annotation_coordinate_type = annotation_coordinate_type
        self._graphic_type = graphic_type
        self._output_dir = output_dir
        self._errors = []


    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Parse CSV to graphic data.

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
                Identifier for each of the polygons. The identifier is a consecutive number 
                going over the whole dataset (not only a single slide). 

        Returns
        -------
        data: dict[str, Any]
            Output data packed into a dict. This will contain the same keys as
            the input dictionary, plus the following additional keys:

            - ann_dcm: hd.ann.MicroscopyBulkSimpleAnnotations:
                DICOM bulk microscopy annotation encoding the original
                annotations in vector format.

        """

        # Unpack inputs
        slide_id = data['slide_id']
        source_image = data['source_image']
        graphic_data = data['graphic_data']
        identifiers = data['identifiers']

        start_time = time()

        logging.info(f'Creating annotation for slide: {slide_id}')

        try:
            ann_dcm = create_bulk_annotations(
                source_image_metadata=source_image,
                graphic_data=graphic_data,
                identifier=identifiers,
                annotation_coordinate_type=self._annotation_coordinate_type,
                graphic_type=self._graphic_type,
            )
        except Exception as e:
            logging.error(f"Error {str(e)}")
            self._errors.append(
                {
                    'slide_id': data['slide_id'],
                    'error_message': str(e),
                    'datetime': str(datetime.datetime.now()),
                }
            )
            errors_df = pd.DataFrame(self._errors)
            errors_df.to_csv(self._output_dir / 'annotation_creator_error_log.csv')
            return None

        stop_time = time()
        duration = stop_time - start_time
        logging.info(
            f'Created annotation for for slide {slide_id} in {duration:.2f}s'
        )

        data['ann_dcm'] = ann_dcm

        # Save some memory
        del data['graphic_data']
        del data['identifiers']

        return data


class AnnotationSaver:

    def __init__(
        self,
        output_dir: Path,
    ):
        """

        Parameters
        ----------
        output_dir: pathlib.Path
            A local output directory to store the downloaded files.

        """

        self._output_dir = output_dir
        self._errors = []

    def __call__(
        self,
        data: dict[str, Any],
    ) -> None:
        """Store files.

        Parameters
        ----------
        data: dict[str, Any]
            Input data packed into a dictionary, containing at least:

            - slide_id: str
                Slide ID for the case.
            - ann_dcm: highdicom.ann.MicroscopyBulkSimpleAnnotations
                Bulk Annotation DICOM object.

        """
        image_start_time = time()

        # Unpack inputs
        slide_id = data['slide_id']
        ann_dcm = data['ann_dcm']

        logging.info(f'Saving annotations for slide {slide_id}')
        
        slide_ann_dir = self._output_dir / slide_id
        slide_ann_dir.mkdir(exist_ok=True)

        try:
            ann_path = f'{slide_ann_dir}/{slide_id}_ann_{suffix}.dcm'
            logging.info(f'Writing annotation to {str(ann_path)}.')
            ann_dcm.save_as(ann_path)

            image_stop_time = time()
            time_for_image = image_stop_time - image_start_time
            logging.info(
                f'Saved annotations for slide {slide_id} in {time_for_image:.2f}s'
            )
            
        except Exception as e:
            logging.error(f"Error {str(e)}")
            self._errors.append(
                {
                    'slide_id': data['slide_id'],
                    'error_message': str(e),
                    'datetime': str(datetime.datetime.now()),
                }
            )
            errors_df = pd.DataFrame(self._errors)
            errors_df.to_csv(self._output_dir / 'upload_error_log.csv')
            return None


def run(
    csv_annotations: Path, 
    csv_rois: Path, 
    output_dir: Path,
    graphic_type: str,
    annotation_coordinate_type: str,
    keep_existing: bool = False,
    workers: int = 0,
    pull_process: bool = True, #?
): 
    logging.basicConfig(level=logging.INFO)

    # Suppress highdicom logging (very talkative)
    logging.getLogger('highdicom.base').setLevel(logging.WARNING)
    logging.getLogger('highdicom.seg.sop').setLevel(logging.WARNING)
    
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)
    
    operations = []
    # first collect metadata 

    parser_kwargs = dict(
        annotation_coordinate_type=annotation_coordinate_type,
        graphic_type=graphic_type,
        workers=workers,
    )
    operations.append((CSVParser, [], parser_kwargs))
    annotation_creator_kwargs = dict(
        annotation_coordinate_type=annotation_coordinate_type,
        graphic_type=graphic_type,
    )
    operations.append((AnnotationCreator, [], annotation_creator_kwargs))
    upload_kwargs = dict(
    )
    operations.append((FileUploader, [], upload_kwargs))
    #pipeline = Pipeline(
    #    operations,
    #    same_process=not pull_process,
    #)
    #pipeline(to_process)


if __name__ == "__main__":
    sim = get_source_image_metadata(Path('/home/dschacherer/bmdeep_conversion/data/bmdeep_DICOM_converted/E2C0BE24560D78C5E599C2A9C9D0BBD2_1_bm'))
    res = preprocess_annotation_csvs(Path('/home/dschacherer/bmdeep_conversion/data/cells.csv'),
                               Path('/home/dschacherer/bmdeep_conversion/data/rois.csv'))
    print(filter_cell_annotations(res, 'F7177163C833DFF4B38FC8D2872F1EC6_1_bm'))