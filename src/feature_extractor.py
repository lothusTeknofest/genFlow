"""
GEN FLOW — Synonymous Mutation Modeling
Feature Engineering Module to extract RSCU, CAI, mRNA stability, conservation, and splicing features.
"""

from __future__ import annotations

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Standard Human Codon Usage Frequencies per thousand (Kazusa Database)
CODON_FREQS = {
    'GCU': 18.4, 'GCC': 27.7, 'GCA': 15.8, 'GCG': 7.4,  # Ala
    'CGU': 4.5, 'CGC': 10.4, 'CGA': 6.2, 'CGG': 11.4, 'AGA': 12.1, 'AGG': 12.0,  # Arg
    'AAU': 17.0, 'AAC': 19.1,  # Asn
    'GAU': 21.8, 'GAC': 25.1,  # Asp
    'UGU': 10.6, 'UGC': 12.6,  # Cys
    'CAA': 12.3, 'CAG': 34.2,  # Gln
    'GAA': 29.0, 'GAG': 39.6,  # Glu
    'GGU': 10.8, 'GGC': 22.2, 'GGA': 16.5, 'GGG': 16.5,  # Gly
    'CAU': 10.9, 'CAC': 15.1,  # His
    'AUU': 16.0, 'AUC': 20.8, 'AUA': 7.5,  # Ile
    'UUA': 7.7, 'UUG': 12.9, 'CUU': 13.2, 'CUC': 19.6, 'CUA': 7.2, 'CUG': 39.6,  # Leu
    'AAA': 24.4, 'AAG': 31.9,  # Lys
    'AUG': 22.0,  # Met
    'UUU': 17.6, 'UUC': 20.3,  # Phe
    'CCU': 17.5, 'CCC': 19.8, 'CCA': 16.9, 'CCG': 6.9,  # Pro
    'UCU': 15.2, 'UCC': 17.7, 'UCA': 12.2, 'UCG': 4.4, 'AGU': 12.1, 'AGC': 19.5,  # Ser
    'ACU': 13.1, 'ACC': 18.9, 'ACA': 15.1, 'ACG': 6.1,  # Thr
    'UGG': 13.2,  # Trp
    'UAU': 12.2, 'UAC': 15.3,  # Tyr
    'GUU': 11.0, 'GUC': 14.5, 'GUA': 7.1, 'GUG': 28.1,  # Val
    'UAA': 1.0, 'UAG': 0.8, 'UGA': 1.6  # Stop
}

CODON_TO_AA = {
    'GCU':'A', 'GCC':'A', 'GCA':'A', 'GCG':'A',
    'CGU':'R', 'CGC':'R', 'CGA':'R', 'CGG':'R', 'AGA':'R', 'AGG':'R',
    'AAU':'N', 'AAC':'N', 'GAU':'D', 'GAC':'D', 'UGU':'C', 'UGC':'C',
    'CAA':'Q', 'CAG':'Q', 'GAA':'E', 'GAG':'E', 'GGU':'G', 'GGC':'G',
    'GGA':'G', 'GGG':'G', 'CAU':'H', 'CAC':'H', 'AUU':'I', 'AUC':'I',
    'AUA':'I', 'UUA':'L', 'UUG':'L', 'CUU':'L', 'CUC':'L', 'CUA':'L',
    'CUG':'L', 'AAA':'K', 'AAG':'K', 'AUG':'M', 'UUU':'F', 'UUC':'F',
    'CCU':'P', 'CCC':'P', 'CCA':'P', 'CCG':'P', 'UCU':'S', 'UCC':'S',
    'UCA':'S', 'UCG':'S', 'AGU':'S', 'AGC':'S', 'ACU':'T', 'ACC':'T',
    'ACA':'T', 'ACG':'T', 'UGG':'W', 'UAU':'Y', 'UAC':'Y', 'GUU':'V',
    'GUC':'V', 'GUA':'V', 'GUG':'V', 'UAA':'*', 'UAG':'*', 'UGA':'*'
}

# Group codons by AA
AA_TO_CODONS: Dict[str, List[str]] = {}
for codon, aa in CODON_TO_AA.items():
    AA_TO_CODONS.setdefault(aa, []).append(codon)

RSCU_TABLE: Dict[str, float] = {}
CAI_TABLE: Dict[str, float] = {}

for aa, codons in AA_TO_CODONS.items():
    total_freq = sum(CODON_FREQS[c] for c in codons)
    max_freq = max(CODON_FREQS[c] for c in codons)
    num_codons = len(codons)
    for c in codons:
        expected = total_freq / num_codons
        RSCU_TABLE[c] = CODON_FREQS[c] / expected if expected > 0 else 0.0
        CAI_TABLE[c] = CODON_FREQS[c] / max_freq if max_freq > 0 else 0.0

# Stacking energies for RNA secondary structure dinucleotides (kcal/mol at 37C)
STACKING_ENERGIES = {
    'AA': -0.9, 'AC': -2.2, 'AG': -2.1, 'AU': -1.1,
    'CA': -2.1, 'CC': -3.3, 'CG': -2.4, 'CU': -2.1,
    'GA': -2.1, 'GC': -3.4, 'GG': -3.3, 'GU': -2.2,
    'UA': -1.3, 'UC': -2.1, 'UG': -2.1, 'UU': -0.9
}

# Ensembl Gene IDs map
GENE_IDS = {
    'BRCA1': 'ENSG00000012048',
    'BRCA2': 'ENSG00000139618',
    'CFTR': 'ENSG00000001626',
    'MAPT': 'ENSG00000186868',
    'PAH': 'ENSG00000171759'
}


class FeatureExtractor:
    def __init__(self):
        self.gene_seqs: Dict[str, str] = {}
        self.gene_exons: Dict[str, List[Dict[str, int]]] = {}
        self.gene_transcripts: Dict[str, str] = {}
        self.gene_metadata: Dict[str, Dict[str, Any]] = {}
        
    def _fetch_gene_data(self, gene: str):
        """Fetches genomic sequence and transcript structure for a target gene once."""
        if gene in self.gene_seqs:
            return
            
        gene_id = GENE_IDS.get(gene)
        if not gene_id:
            return
            
        # 1. Fetch sequence
        print(f"Fetching genomic sequence for {gene}...")
        url_seq = f"https://rest.ensembl.org/sequence/id/{gene_id}?content-type=application/json"
        try:
            r = requests.get(url_seq)
            if r.status_code == 200:
                data = r.json()
                self.gene_seqs[gene] = data['seq']
                desc = data.get('desc', '')
                # Parse: chromosome:GRCh38:17:43044292:43170245:-1
                parts = desc.split(':')
                if len(parts) >= 6:
                    self.gene_metadata[gene] = {
                        'assembly': parts[1],
                        'chrom': f"chr{parts[2]}" if not parts[2].startswith('chr') else parts[2],
                        'start': int(parts[3]),
                        'end': int(parts[4]),
                        'strand': int(parts[5])
                    }
        except Exception as e:
            print(f"Error fetching sequence for {gene}: {e}")
            
        # 2. Fetch transcript structure (exons of canonical transcript)
        print(f"Fetching transcript structure for {gene}...")
        url_tx = f"https://rest.ensembl.org/lookup/id/{gene_id}?expand=1;content-type=application/json"
        try:
            r = requests.get(url_tx)
            if r.status_code == 200:
                transcripts = r.json().get('Transcript', [])
                canonical = [t for t in transcripts if t.get('is_canonical') == 1]
                if canonical:
                    ct = canonical[0]
                    self.gene_transcripts[gene] = ct['id']
                    self.gene_exons[gene] = sorted(
                        [{'start': e['start'], 'end': e['end']} for e in ct.get('Exon', [])],
                        key=lambda x: x['start']
                    )
        except Exception as e:
            print(f"Error fetching transcript for {gene}: {e}")

    def get_local_sequence(self, gene: str, position: int) -> Tuple[str, str]:
        """Gets local 21nt wildtype and mutant sequence window centered on position."""
        self._fetch_gene_data(gene)
        
        seq = self.gene_seqs.get(gene)
        meta = self.gene_metadata.get(gene)
        
        if not seq or not meta:
            return "", ""
            
        strand = meta['strand']
        if strand == 1:
            start_coord = meta['start']
            offset = position - start_coord
        else:
            end_coord = meta['end']
            offset = end_coord - position
            
        # Make sure window stays inside sequence boundaries
        if offset < 10 or offset >= len(seq) - 10:
            return "", ""
            
        wt_window = seq[offset - 10 : offset + 11].upper()
        return wt_window, meta['chrom']

    def calculate_stacking_energy(self, rna_seq: str) -> float:
        """Sums the stacking free energy of adjacent nucleotides in RNA sequence."""
        rna_seq = rna_seq.replace('T', 'U')
        energy = 0.0
        for i in range(len(rna_seq) - 1):
            dinuc = rna_seq[i:i+2]
            energy += STACKING_ENERGIES.get(dinuc, 0.0)
        return energy

    def extract_features_for_variant(self, row: pd.Series) -> Dict[str, Any]:
        """Extracts RSCU, CAI, mRNA stability, conservation, and splicing features for a variant."""
        gene = row['Gene']
        chrom = row['Chromosome']
        pos = int(row['Position'])
        ref = str(row['Ref']).upper()
        alt = str(row['Alt']).upper()
        
        self._fetch_gene_data(gene)
        
        features = {
            'Variant_ID': row['Variant_ID'],
            'Gene': gene,
            'Chromosome': chrom,
            'Position': pos,
            'Ref': ref,
            'Alt': alt,
            'Label': row['Label'],
            'RSCU_WT': 0.0,
            'RSCU_MUT': 0.0,
            'Delta_RSCU': 0.0,
            'CAI_WT': 0.0,
            'CAI_MUT': 0.0,
            'Delta_CAI': 0.0,
            'wt_MFE': 0.0,
            'mut_MFE': 0.0,
            'Delta_Delta_G': 0.0,
            'phyloP': 0.0,
            'phastCons': 0.0,
            'splicing_dist_to_junction': 9999.0
        }
        
        # 1. Fetch Splicing Distance to Junction
        exons = self.gene_exons.get(gene, [])
        if exons:
            min_dist = min(min(abs(pos - e['start']), abs(pos - e['end'])) for e in exons)
            features['splicing_dist_to_junction'] = min_dist
            
        # 2. Fetch Codon change & consequence from Ensembl VEP
        tx_id = self.gene_transcripts.get(gene)
        if tx_id:
            url_vep = f"https://rest.ensembl.org/vep/human/region/{chrom.replace('chr', '')}:{pos}-{pos}/{alt}?content-type=application/json"
            try:
                r = requests.get(url_vep)
                if r.status_code == 200:
                    data = r.json()
                    t_consequences = data[0].get('transcript_consequences', [])
                    
                    # Find canonical transcript consequence, or fallback to first synonymous consequence
                    selected_conseq = None
                    for tc in t_consequences:
                        if tc.get('transcript_id') == tx_id and 'synonymous_variant' in tc.get('consequence_terms', []):
                            selected_conseq = tc
                            break
                    if not selected_conseq:
                        for tc in t_consequences:
                            if 'synonymous_variant' in tc.get('consequence_terms', []):
                                selected_conseq = tc
                                break
                                
                    if selected_conseq and 'codons' in selected_conseq:
                        codons = selected_conseq['codons'].upper().split('/')
                        if len(codons) == 2:
                            wt_codon = codons[0]
                            mut_codon = codons[1]
                            
                            features['RSCU_WT'] = RSCU_TABLE.get(wt_codon, 0.0)
                            features['RSCU_MUT'] = RSCU_TABLE.get(mut_codon, 0.0)
                            features['Delta_RSCU'] = features['RSCU_MUT'] - features['RSCU_WT']
                            
                            features['CAI_WT'] = CAI_TABLE.get(wt_codon, 0.0)
                            features['CAI_MUT'] = CAI_TABLE.get(mut_codon, 0.0)
                            features['Delta_CAI'] = features['CAI_MUT'] - features['CAI_WT']
            except Exception as e:
                pass
                
        # 3. Calculate mRNA Folding Energy Change (Delta Delta G)
        wt_seq, resolved_chrom = self.get_local_sequence(gene, pos)
        if wt_seq and len(wt_seq) == 21:
            meta = self.gene_metadata[gene]
            strand = meta['strand']
            
            # Transcript allele
            tx_alt = alt
            if strand == -1:
                # Complement of alternate allele if on minus strand
                comp_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
                tx_alt = comp_map.get(alt, alt)
                
            # Replace wildtype nucleotide (middle of 21nt window, index 10) with mutant
            mut_seq = wt_seq[:10] + tx_alt + wt_seq[11:]
            
            wt_energy = self.calculate_stacking_energy(wt_seq)
            mut_energy = self.calculate_stacking_energy(mut_seq)
            
            features['wt_MFE'] = wt_energy
            features['mut_MFE'] = mut_energy
            features['Delta_Delta_G'] = mut_energy - wt_energy
            
        # 4. Fetch Conservation scores from UCSC Genome Browser track API
        try:
            url_phyloP = f"https://api.genome.ucsc.edu/getData/track?genome=hg38&track=phyloP100way&chrom={chrom}&start={pos-1}&end={pos}"
            r = requests.get(url_phyloP)
            if r.status_code == 200:
                vals = r.json().get('phyloP100way', [])
                if vals:
                    features['phyloP'] = vals[0].get('value', 0.0)
        except Exception:
            pass
            
        try:
            url_phastCons = f"https://api.genome.ucsc.edu/getData/track?genome=hg38&track=phastCons100way&chrom={chrom}&start={pos-1}&end={pos}"
            r = requests.get(url_phastCons)
            if r.status_code == 200:
                vals = r.json().get('phastCons100way', [])
                if vals:
                    features['phastCons'] = vals[0].get('value', 0.0)
        except Exception:
            pass
            
        return features


if __name__ == "__main__":
    input_path = DATA_DIR / "clinvar_synonymous.csv"
    if not input_path.exists():
        print(f"Error: {input_path} not found. Please run data_fetcher.py first.")
        sys.exit(1)
        
    df = pd.read_csv(input_path)
    # limit to process quickly (e.g. 500 variants to complete training fast)
    df_sample = df.sample(n=min(500, len(df)), random_state=42).copy()
    
    print(f"Extracting features for {len(df_sample)} variants...")
    extractor = FeatureExtractor()
    
    feature_list = []
    for idx, row in df_sample.iterrows():
        if idx % 50 == 0:
            print(f"Progress: {idx}/{len(df_sample)}...")
        # Avoid hitting API rate limits
        time.sleep(0.1)
        feat = extractor.extract_features_for_variant(row)
        feature_list.append(feat)
        
    df_features = pd.DataFrame(feature_list)
    output_path = DATA_DIR / "processed_clinvar_synonymous.csv"
    df_features.to_csv(output_path, index=False)
    print(f"Saved processed features to {output_path}")
