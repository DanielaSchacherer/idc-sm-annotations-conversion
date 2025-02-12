import highdicom as hd
import pydicom
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code

#from idc_annotation_conversion.git_utils import (
#    get_git_remote_url,
#    get_git_commit_hash,
#)

# Basic Metadata
manufacturer = 'Fraunhofer MEVIS'
manufacturer_model_name = 'BMDeep'
software_versions = 'tbd' #get_git_remote_url(simplify=True)
device_serial_number = 'tbd' #get_git_commit_hash()

# Labels description
labels = ['neutrophilic_metamyelocyte']
finding_categories = [Code('91723000', 'SCT', 'Anatomical Stucture')]
finding_types = [Code('118957004', 'SCT', 'Normal cell')]
labels_dict = dict(
    label=labels, 
    finding_category=finding_categories, 
    finding_type=finding_types
)

# Algorithm Identification
algorithm_identification = hd.AlgorithmIdentificationSequence(
    name='Fraunhofer MEVIS BMDeep',
    version='1.0',
    source='tbd',
    family=codes.cid7162.ArtificialIntelligence,
)

# DOI of the conversion page in Zenodo for other clinical trial protocol
# These are not yet in pydicom's data dict so we have to set the elements
# manually
# IssuerOfClinicalTrialProtocolID
issuer_tag = pydicom.tag.Tag(0x0012, 0x0022)
# OtherClinicalTrialProtocolIDsSequence
other_trials_seq_tag = pydicom.tag.Tag(0x0012, 0x0023)

clinical_trial_ids_item = pydicom.Dataset()
issuer_value = "DOI"
clinical_trial_ids_item.add(
    pydicom.DataElement(
        issuer_tag,
        "LO",
        issuer_value,
    )
)
clinical_trial_ids_item.ClinicalTrialProtocolID = "doi:10.5281/zenodo.11099005"
other_trials_seq_element = pydicom.DataElement(
    other_trials_seq_tag,
    "SQ",
    [clinical_trial_ids_item],
)
