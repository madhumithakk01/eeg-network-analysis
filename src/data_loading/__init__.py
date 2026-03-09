"""EEG and metadata loading; channel consistency analysis."""

# Channel consistency stage (see docs/channel_consistency_design.md):
# - patient_list: load patient IDs from canonical or split CSV
# - hea_parsing: extract channel names from WFDB .hea headers
# - channel_filter: exclude non-EEG channels (EKG, EMG, etc.)
# - channel_inventory: orchestrate scan, frequency, intersection, write outputs
