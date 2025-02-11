"""Entrypoint for Pan Cancer Nuclei Segmentation annotation conversion."""
import os
import datetime
import pydicom 
import logging
from pathlib import Path
from time import time
from typing import Any, Dict, Optional

import highdicom as hd
import pandas as pd

#from idc_annotation_conversion.pan_cancer_nuclei_seg.convert import (
#    get_graphic_data,
#    create_bulk_annotations,
#)
#from idc_annotation_conversion.mp_utils import Pipeline

ANNOTATION_PREFIX = 'cnn-nuclear-segmentations-2019/data-files/'

# could be written as class as well
def preprocess_annotation_csvs(annotation_csv: Path, roi_csv: Path) -> pd.DataFrame: 
    annotations = pd.read_csv(annotation_csv)
    rois = pd.read_csv(roi_csv)
    print(annotations.head())
    print(rois.head())

def filter_annotations(annotations: pd.DataFrame, slide_id: str) -> pd.DataFrame: 
    pass 

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


def get_ann_file_name(
    output_root: str,
    slide_id: str, 
    suffix: str
) -> str:
    """Get the name of the file where an annotation should be stored."""
    return Path(f'{output_root}{slide_id}/{slide_id}_ann_{suffix}.dcm')


class AnnotationParser:

    """Class that parses CSV annotations to graphic data."""

    def __init__(
        self,
        annotation_coordinate_type: str,
        graphic_type: str
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

        """
        self._annotation_coordinate_type = annotation_coordinate_type
        self._graphic_type = graphic_type
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
                annotation_csvs='tbd',
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
            errors_df.to_csv('conversion_error_log.csv')
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

        """
        self._annotation_coordinate_type = annotation_coordinate_type
        self._graphic_type = graphic_type
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
            errors_df.to_csv("annotation_creator_error_log.csv")
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


# FileSaver
class FileUploader:

    def __init__(
        self,
        output_bucket_obj: Optional[storage.Bucket],
        output_blob_root: str,
        output_dir: Optional[Path],
        dicom_archive: Optional[str] = None,
        archive_token_url: Optional[str] = None,
        archive_client_id: Optional[str] = None,
        archive_client_secret: Optional[str] = None,
    ):
        """

        Parameters
        ----------
        output_bucket_obj: Optional[google.storage.Bucket]
            Output bucket, if storing in a bucket.
        output_dir: Optional[pathlib.Path]
            A local output directory to store a copy of the downloaded files
            in, if required.
        dicom_archive: Optional[str], optional
            Additionally store images to this DICOM archive.
        archive_token_url: Optional[str], optional
            URL to use to request an OAuth token to access the archive.,
        archive_client_id: Optional[str], optional
            Client ID to use for OAuth token request.,
        archive_client_secret: Optional[str], optional
            Client secret to use for OAuth token request. If none, user will
            be prompted for secret.

        """
        self._output_bucket_obj = output_bucket_obj
        self._output_blob_root = output_blob_root
        self._output_dir = output_dir
        self._dicom_archive = dicom_archive
        self._archive_token_url = archive_token_url
        self._archive_client_id = archive_client_id
        self._archive_client_secret = archive_client_secret
        self._errors = []

    def __call__(
        self,
        data: dict[str, Any],
    ) -> None:
        """Upload files.

        Parameters
        ----------
        data: dict[str, Any]
            Input data packed into a dictionary, containing at least:

            - collection: str
                Collection name for the case.
            - container_id: str
                Container ID for the case.
            - ann_dcm: highdicom.ann.MicroscopyBulkSimpleAnnotations
                Bulk Annotation DICOM object.
            - seg_dcm: Optional[list[highdicom.seg.Segmentation]]
                Converted segmentations, if required.

        """
        image_start_time = time()

        # Unpack inputs
        collection = data["collection"]
        container_id = data["container_id"]
        ann_dcm = data["ann_dcm"]
        seg_dcms = data.get("seg_dcms")

        logging.info(f"Uploading annotations for {container_id}")

        if self._output_dir is not None:
            collection_dir = self._output_dir / collection
            collection_dir.mkdir(exist_ok=True)

        try:
            # Store objects to filesystem
            if self._output_dir is not None:
                ann_path = collection_dir / f"{container_id}_ann.dcm"

                logging.info(f"Writing annotation to {str(ann_path)}.")
                ann_dcm.save_as(ann_path)

                if seg_dcms is not None:
                    for s, seg_dcm in enumerate(seg_dcms):
                        seg_path = (
                            collection_dir / f"{container_id}_seg_{s}.dcm"
                        )
                        logging.info(
                            f"Writing segmentation to {str(seg_path)}."
                        )
                        seg_dcm.save_as(seg_path)

            image_stop_time = time()
            time_for_image = image_stop_time - image_start_time
            logging.info(
                f"Uploaded annotations for {container_id} in "
                f"{time_for_image:.2f}s"
            )
            
        except Exception as e:
            logging.error(f"Error {str(e)}")
            self._errors.append(
                {
                    "collection": collection,
                    "container_id": container_id,
                    "error_message": str(e),
                    "datetime": str(datetime.datetime.now()),
                }
            )
            errors_df = pd.DataFrame(self._errors)
            errors_df.to_csv("upload_error_log.csv")
            return None


@click.command()
@click.option(
    "-c",
    "--collections",
    multiple=True,
    type=click.Choice(COLLECTIONS),
    help="Collections to use, all by default.",
    show_choices=True,
)
@click.option(
    "-l",
    "--csv-blob",
    help=(
        "Specify a single CSV blob to process, using its path within "
        "the bucket."
    ),
)
@click.option(
    "-f",
    "--blob-filter",
    help=(
        "Only process annotations blobs whose name contains this string."
    ),
)
@click.option(
    "--number",
    "-n",
    type=int,
    help="Number of annotations to process. All by default.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path, file_okay=False),
    help="Output directory, default: no output directory",
)
@click.option(
    "--output-bucket",
    "-b",
    help="Output bucket",
    show_default=True,
)
@click.option(
    "--store-bucket/--no-store-bucket",
    "-k/-K",
    help="Whether to store outputs to the bucket in this run.",
    default=True,
    show_default=True,
)
@click.option(
    "--keep-existing/--overwrite-existing",
    "-m/-M",
    help="Only process a case if the output does not exist in the bucket.",
    default=False,
    show_default=True,
)
@click.option(
    "--output-prefix",
    "-p",
    help="Prefix for all output blobs. Default no prefix.",
)
@click.option(
    "--graphic-type",
    "-g",
    default="POLYGON",
    type=click.Choice(
        [v.name for v in hd.ann.GraphicTypeValues],
        case_sensitive=False,
    ),
    show_default=True,
    help=(
        "Graphic type to use to store all nuclei. Note that all "
        "but 'POLYGON' result in simplification and loss of "
        "information in the annotations."
    ),
)
@click.option(
    "--store-wsi-dicom/--no-store-wsi-dicom",
    "-d/-D",
    default=False,
    show_default=True,
    help=(
        "Download all WSI DICOM files and store in the output directory "
        "(if any)."
    ),
)
@click.option(
    "--annotation-coordinate-type",
    "-a",
    type=click.Choice(
        [v.name for v in hd.ann.AnnotationCoordinateTypeValues],
        case_sensitive=False,
    ),
    default="SCOORD",
    show_default=True,
    help=(
        "Coordinate type for points stored in the microscopy annotations. "
        "SCOORD: 2D, SCOORD3D: 3D."
    ),
)
@click.option(
    "--dimension-organization-type",
    "-T",
    type=click.Choice(
        [v.name for v in hd.DimensionOrganizationTypeValues],
        case_sensitive=False,
    ),
    default="TILED_FULL",
    show_default=True,
    help=(
        "Dimension organization type for segmentations. TILED_FULL (default) "
        "or TILED_SPARSE."
    ),
)
@click.option(
    "--with-segmentation/--without-segmentation",
    "-s/-S",
    default=True,
    show_default=True,
    help="Include a segmentation image in the output.",
)
@click.option(
    "--create-pyramid/--no-create-pyramid",
    "-q/-Q",
    default=True,
    show_default=True,
    help="Create a full segmentation pyramid series.",
)
@click.option(
    "--segmentation-type",
    "-t",
    type=click.Choice(
        [v.name for v in hd.seg.SegmentationTypeValues],
        case_sensitive=False,
    ),
    default="BINARY",
    show_default=True,
    help="Segmentation type for the Segmentation Image, if any.",
)
@click.option(
    "--dicom-archive",
    "-x",
    help="Additionally store outputs to this DICOM archive.",
)
@click.option(
    "--archive-token-url",
    "-u",
    help="URL to use to request an OAuth token to access the archive.",
)
@click.option(
    "--archive-client-id",
    "-i",
    help="Client ID to use for OAuth token request.",
)
@click.option(
    "--archive-client-secret",
    "-y",
    help=(
        "Client secret to use for OAuth token request. If none, user will "
        "be prompted for secret."
    )
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=0,
    help=(
        "Number of subprocesses for CSV parsing to use. If 0, the main thread "
        "is used."
    ),
)
@click.option(
    "--pull-process/--no-pull-process",
    "-r/-R",
    default=True,
    show_default=True,
    help="Use a separate process to pull images.",
)


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
    logging.getLogger("highdicom.base").setLevel(logging.WARNING)
    logging.getLogger("highdicom.seg.sop").setLevel(logging.WARNING)
    
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
    pipeline = Pipeline(
        operations,
        same_process=not pull_process,
    )
    pipeline(to_process)


if __name__ == "__main__":
    sim = get_source_image_metadata(Path('/home/dschacherer/bmdeep_conversion/data/bmdeep_DICOM_converted/'))
    print(sim)
    preprocess_annotation_csvs(Path('/home/dschacherer/bmdeep_conversion/data/cells.csv'),
                               Path('/home/dschacherer/bmdeep_conversion/data/rois.csv'))