import os
import re
import argparse
import pandas as pd
from Bio import SeqIO
from collections import defaultdict

################################################
#              FIMOsa                          #
################################################

# Takes FIMO .tsv outputs and pulls annotation information from .gbff files; 
# in short, helps identify what genes are near FIMO motif matches.
# ─────────────────────────────────────────────
# ANNOTATION LOOKUP
# ─────────────────────────────────────────────

def build_annotation_lookup(gbff_file):
    lookup = {}
    for genome in SeqIO.parse(gbff_file, "genbank"):
        for feat in genome.features:
            if feat.type == "CDS":
                locus = feat.qualifiers.get("locus_tag", [None])[0]
                product = feat.qualifiers.get("product", ["hypothetical protein"])[0]
                accession = feat.qualifiers.get("protein_id", [None])[0]
                if locus:
                    lookup[locus] = {
                        "product": product,
                        "accession": accession,
                        "strand": feat.location.strand,
                        "gene_start": int(feat.location.start) + 1,
                        "gene_end": int(feat.location.end)
                    }
    return lookup

# ─────────────────────────────────────────────
# GENOMIC COORDINATE CALCULATION
# ─────────────────────────────────────────────

def compute_genomic_coords(row, lookup):
    locus = row["sequence_name"]
    if locus not in lookup:
        return pd.Series({"genomic_start": None, "genomic_stop": None})

    info = lookup[locus]
    strand = info["strand"]
    motif_start = row["start"]
    motif_stop = row["stop"]

    if strand == 1:
        window_origin = info["gene_start"] - 300
        genomic_start = window_origin + motif_start
        genomic_stop = window_origin + motif_stop
    else:
        window_origin = info["gene_end"]
        genomic_stop = window_origin - motif_start
        genomic_start = window_origin - motif_stop

    return pd.Series({"genomic_start": int(genomic_start), "genomic_stop": int(genomic_stop)})

# ─────────────────────────────────────────────
# LOCUS PREFIX EXTRACTION
# ─────────────────────────────────────────────

def get_locus_prefix(fimo_tsv):
    """Extract locus prefix (e.g. 'ABEA36_') from sequence_name column."""
    try:
        df = pd.read_csv(fimo_tsv, sep="\t", comment="#")
        df = df.dropna(subset=["sequence_name"])
        if df.empty:
            return None
        # extract prefix: everything up to and including the underscore before RS
        sample = df["sequence_name"].iloc[0]
        match = re.match(r"^([A-Z0-9]+_)", sample)
        if match:
            return match.group(1)
    except Exception as e:
        print(f"  WARNING: could not extract prefix from {fimo_tsv}: {e}")
    return None

def find_matching_gbff(prefix, gbff_dir):
    """Find a .gbff file whose locus tags match the given prefix."""
    for fname in os.listdir(gbff_dir):
        if not fname.endswith(".gbff"):
            continue
        fpath = os.path.join(gbff_dir, fname)
        # quick scan for the prefix in locus tags
        for genome in SeqIO.parse(fpath, "genbank"):
            for feat in genome.features:
                if feat.type == "CDS":
                    locus = feat.qualifiers.get("locus_tag", [None])[0]
                    if locus and locus.startswith(prefix):
                        return fpath
    return None

# ─────────────────────────────────────────────
# PARSE SINGLE FIMO FILE
# ─────────────────────────────────────────────

def parse_fimo(fimo_tsv, gbff_file, qvalue_thresh, top_n):
    df = pd.read_csv(fimo_tsv, sep="\t", comment="#")
    df = df.dropna(subset=["q-value"])

    hits = df[df["q-value"] < qvalue_thresh].sort_values("score", ascending=False)
    hits = hits.head(top_n)

    if hits.empty:
        return hits

    lookup = build_annotation_lookup(gbff_file)
    hits = hits.copy()

    # append new columns
    hits["locus_id"] = hits["sequence_name"]
    hits["cds_start"] = hits["sequence_name"].map(lambda x: lookup.get(x, {}).get("gene_start"))
    hits["cds_stop"] = hits["sequence_name"].map(lambda x: lookup.get(x, {}).get("gene_end"))
    hits["protein_accession"] = hits["sequence_name"].map(lambda x: lookup.get(x, {}).get("accession"))
    hits["product"] = hits["sequence_name"].map(lambda x: lookup.get(x, {}).get("product"))

    # genomic coordinates
    genomic_coords = hits.apply(lambda row: compute_genomic_coords(row, lookup), axis=1)
    hits = pd.concat([hits, genomic_coords], axis=1)

    # rename start/stop to be unambiguous
    hits = hits.rename(columns={
        "start": "motif_window_start",
        "stop": "motif_window_stop"
    })

    # final column order: all original FIMO columns first, then appended
    cols = [
        "motif_id", "motif_alt_id", "sequence_name",
        "motif_window_start", "motif_window_stop", "strand",
        "score", "p-value", "q-value", "matched_sequence",
        "locus_id", "cds_start", "cds_stop",
        "protein_accession", "product",
        "genomic_start", "genomic_stop"
    ]
    cols = [c for c in cols if c in hits.columns]
    return hits[cols]

# ─────────────────────────────────────────────
# CROSS-GENOME COMPARISON
# ─────────────────────────────────────────────

def build_comparison(all_hits, min_genomes):
    """
    Build a summary of products appearing in >= min_genomes genomes.
    all_hits: dict of {genome_name: dataframe}
    """
    product_counts = defaultdict(list)

    for genome_name, df in all_hits.items():
        if df.empty:
            continue
        for product in df["product"].dropna().unique():
            product_counts[product].append(genome_name)

    rows = []
    for product, genomes in product_counts.items():
        if len(genomes) >= min_genomes:
            rows.append({
                "product": product,
                "n_genomes": len(genomes),
                "genomes": ", ".join(sorted(genomes))
            })

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows).sort_values("n_genomes", ascending=False)
    return summary

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parse FIMO outputs across multiple genomes, annotate with gbff data, and identify conserved hits."
    )
    parser.add_argument("--fimo_dir", required=True,
                        help="Directory containing fimo .tsv files")
    parser.add_argument("--gbff_dir", required=True,
                        help="Directory containing .gbff genome files")
    parser.add_argument("--outdir", required=True,
                        help="Output directory for annotated TSVs and summary")
    parser.add_argument("--qvalue", type=float, default=0.005,
                        help="q-value threshold (default: 0.005)")
    parser.add_argument("--top", type=int, default=50,
                        help="Max hits per genome to retain (default: 50)")
    parser.add_argument("--min_genomes", type=int, default=2,
                        help="Min number of genomes a product must appear in to be flagged as conserved (default: 2)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    fimo_files = [
        os.path.join(args.fimo_dir, f)
        for f in os.listdir(args.fimo_dir)
        if f.endswith(".tsv")
    ]

    if not fimo_files:
        print(f"No .tsv files found in {args.fimo_dir}")
        return

    all_hits = {}
    unmatched = []

    for fimo_tsv in sorted(fimo_files):
        fname = os.path.basename(fimo_tsv)
        print(f"\nProcessing {fname}...")

        prefix = get_locus_prefix(fimo_tsv)
        if not prefix:
            print(f"  WARNING: could not determine locus prefix, skipping.")
            unmatched.append(fname)
            continue

        print(f"  Locus prefix detected: {prefix}")
        gbff_file = find_matching_gbff(prefix, args.gbff_dir)

        if not gbff_file:
            print(f"  WARNING: no matching .gbff found for prefix {prefix}, skipping.")
            unmatched.append(fname)
            continue

        print(f"  Matched to: {os.path.basename(gbff_file)}")

        hits = parse_fimo(fimo_tsv, gbff_file, args.qvalue, args.top)
        print(f"  Hits after q < {args.qvalue} filter and top {args.top}: {len(hits)}")

        # save individual annotated TSV
        genome_name = prefix.rstrip("_")
        out_path = os.path.join(args.outdir, f"{genome_name}_hits_annotated.tsv")
        hits.to_csv(out_path, sep="\t", index=False)
        print(f"  Saved to {out_path}")

        all_hits[genome_name] = hits

    # cross-genome comparison
    print(f"\nBuilding cross-genome comparison (min_genomes={args.min_genomes})...")
    summary = build_comparison(all_hits, args.min_genomes)

    if summary.empty:
        print("  No conserved hits found across genomes at the specified threshold.")
    else:
        summary_path = os.path.join(args.outdir, "conserved_hits_summary.tsv")
        summary.to_csv(summary_path, sep="\t", index=False)
        print(f"  Conserved hits saved to {summary_path}")
        print(f"\n  Top conserved products:")
        print(summary.head(20).to_string(index=False))

    if unmatched:
        print(f"\nWARNING: the following files could not be matched to a .gbff and were skipped:")
        for f in unmatched:
            print(f"  {f}")

    print("\nDone.")

if __name__ == "__main__":
    main()
