"""
GEN FLOW — Synonymous Mutation Modeling
Data Retrieval Module to fetch synonymous variants from ClinVar.
"""

from __future__ import annotations

import os
import sys
import json
import pandas as pd
from types import ModuleType
from pathlib import Path

# Mock Unix-only fcntl module for Windows compatibility
if sys.platform.startswith('win'):
    os.makedirs('C:/tmp', exist_ok=True)
    if 'fcntl' not in sys.modules:
        fcntl_mock = ModuleType('fcntl')
        fcntl_mock.flock = lambda fd, op: None
        fcntl_mock.LOCK_EX = 1
        fcntl_mock.LOCK_SH = 2
        fcntl_mock.LOCK_UN = 4
        fcntl_mock.LOCK_NB = 8
        sys.modules['fcntl'] = fcntl_mock

# Add clinvar skill script path to sys.path
CLINVAR_SKILL_PATH = r"C:\Users\Tuğba\.gemini\config\plugins\science\skills\clinvar_database\scripts"
if CLINVAR_SKILL_PATH not in sys.path:
    sys.path.append(CLINVAR_SKILL_PATH)

from clinvar_api import ClinVarClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TARGET_GENES = ["BRCA1", "BRCA2", "CFTR", "MAPT", "PAH"]


def map_refseq_to_chr(refseq_acc: str) -> str | None:
    """Maps NC_ RefSeq accession to chromosome string (e.g. NC_000017.11 -> chr17)."""
    if not refseq_acc.startswith("NC_"):
        return None
    try:
        # Extract the chromosome number part (e.g. NC_000017 -> 17)
        parts = refseq_acc.split('.')[0].split('_')[1]
        chrom_num = int(parts)
        if chrom_num == 23:
            return "chrX"
        elif chrom_num == 24:
            return "chrY"
        elif 1 <= chrom_num <= 22:
            return f"chr{chrom_num}"
    except (IndexError, ValueError):
        pass
    return None


def fetch_clinvar_synonymous_data(limit_per_class: int = 1000) -> pd.DataFrame:
    """
    Fetches pathogenic and benign synonymous mutations from ClinVar for target genes.
    """
    client = ClinVarClient()
    
    # 1. Build queries
    genes_term = " OR ".join([f"{g}[gene]" for g in TARGET_GENES])
    
    query_pathogenic = (
        f"({genes_term}) AND \"synonymous variant\"[molecular consequence] AND "
        "(pathogenic[clinical significance] OR \"likely pathogenic\"[clinical significance])"
    )
    
    query_benign = (
        f"({genes_term}) AND \"synonymous variant\"[molecular consequence] AND "
        "(benign[clinical significance] OR \"likely benign\"[clinical significance])"
    )
    
    print("Searching ClinVar...")
    print(f"Pathogenic Query: {query_pathogenic[:100]}...")
    print(f"Benign Query: {query_benign[:100]}...")
    
    path_search = client.search_variants(query_pathogenic, retmax=limit_per_class)
    benign_search = client.search_variants(query_benign, retmax=limit_per_class)
    
    all_variants = []
    
    # Process queries: (search_result, label)
    for search_res, label in [(path_search, 1), (benign_search, 0)]:
        v_ids = search_res.get("variant_ids", [])
        if not v_ids:
            continue
            
        print(f"Fetching summaries for {len(v_ids)} variants with label {label}...")
        
        # Batch size for esummary
        batch_size = 200
        for i in range(0, len(v_ids), batch_size):
            batch_ids = v_ids[i:i+batch_size]
            try:
                summaries = client.get_interpretation_summary(batch_ids)
                
                # Fetch raw esummary data to parse SPDI coordinates
                params = {'db': 'clinvar', 'id': ','.join(batch_ids), 'retmode': 'json'}
                response = client._request('esummary.fcgi', params)
                raw_data = json.loads(response.content).get('result', {})
                
                for sum_dict in summaries:
                    var_id = sum_dict['variant_id']
                    raw_var = raw_data.get(var_id, {})
                    
                    # Extract coordinates via SPDI
                    spdi = raw_var.get("canonical_spdi", "")
                    chrom = None
                    pos = None
                    ref = None
                    alt = None
                    
                    if spdi:
                        parts = spdi.split(':')
                        if len(parts) == 4:
                            chrom = map_refseq_to_chr(parts[0])
                            # SPDI is 0-based, convert to 1-based
                            pos = int(parts[1]) + 1
                            ref = parts[2]
                            alt = parts[3]
                            
                    # Fallback to GRCh38 location
                    if not chrom:
                        for loc in raw_var.get("variation_set", [{}])[0].get("variation_loc", []):
                            if loc.get("assembly_name") == "GRCh38":
                                chrom = f"chr{loc.get('chr')}" if not str(loc.get('chr')).startswith('chr') else loc.get('chr')
                                pos = int(loc.get("start")) if loc.get("start") else None
                                break
                                
                    # Get gene symbol
                    gene_symbol = sum_dict.get("genes", [{}])[0].get("symbol", "Unknown")
                    
                    if chrom and pos:
                        all_variants.append({
                            "Variant_ID": f"CV_{var_id}",
                            "ClinVar_ID": var_id,
                            "Gene": gene_symbol,
                            "Chromosome": chrom,
                            "Position": pos,
                            "Ref": ref,
                            "Alt": alt,
                            "Title": sum_dict.get("title", ""),
                            "Label": label
                        })
            except Exception as e:
                print(f"Error fetching batch {i}: {e}")
                
    df = pd.DataFrame(all_variants)
    print(f"Successfully processed {len(df)} variants.")
    return df


if __name__ == "__main__":
    df_syn = fetch_clinvar_synonymous_data(limit_per_class=1000)
    output_path = DATA_DIR / "clinvar_synonymous.csv"
    df_syn.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
