"""
GEN FLOW — Synonymous Mutation Modeling
Tests for the feature extraction module.
"""

from __future__ import annotations

import pytest
import pandas as pd
from src.feature_extractor import FeatureExtractor, RSCU_TABLE, CAI_TABLE, CODON_TO_AA


def test_codon_usage_bias_tables():
    # Methionine (Met, M) only has one codon: AUG. Its RSCU and CAI should be 1.0
    assert CODON_TO_AA['AUG'] == 'M'
    assert RSCU_TABLE['AUG'] == 1.0
    assert CAI_TABLE['AUG'] == 1.0
    
    # Check that RSCU and CAI values exist for other standard codons
    assert 'ACC' in RSCU_TABLE
    assert 'ACC' in CAI_TABLE
    assert RSCU_TABLE['ACC'] > 0
    assert CAI_TABLE['ACC'] > 0


def test_stacking_energy():
    extractor = FeatureExtractor()
    # Test simple single-stranded RNA sequence stacking energy
    seq = "AUG"  # Dinucleotides: AU, UG
    energy = extractor.calculate_stacking_energy(seq)
    
    # Expected: STACKING_ENERGIES['AU'] + STACKING_ENERGIES['UG'] = -1.1 + -2.1 = -3.2
    assert pytest.approx(energy) == -3.2


def test_splicing_distance_calculation():
    extractor = FeatureExtractor()
    
    # Mock some exons
    mock_exons = [
        {'start': 100, 'end': 200},
        {'start': 300, 'end': 400}
    ]
    extractor.gene_exons['TEST_GENE'] = mock_exons
    extractor.gene_transcripts['TEST_GENE'] = 'TEST_TX'
    
    # Test variant inside exon 1
    row = pd.Series({
        'Variant_ID': 'VAR_TEST',
        'Gene': 'TEST_GENE',
        'Chromosome': 'chr1',
        'Position': 150,
        'Ref': 'A',
        'Alt': 'G',
        'Label': 0
    })
    
    # Mock Ensembl VEP API to avoid actual call during unit tests
    extractor._fetch_gene_data = lambda gene: None
    
    feat = extractor.extract_features_for_variant(row)
    
    # Nearest boundary of exon 1 (100 or 200) to position 150:
    # abs(150 - 100) = 50, abs(150 - 200) = 50. Min is 50
    assert feat['splicing_dist_to_junction'] == 50
