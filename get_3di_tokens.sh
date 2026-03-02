#!/bin/bash
FOLDSEEK=/home/zww20/rds/hpc-work/GDL/PiFold-main/foldseek/bin/foldseek
PDB_DIR=/home/zww20/rds/hpc-work/GDL/PiFold-main/data/cath_pdb
OUT_DIR=/home/zww20/rds/hpc-work/GDL/PiFold-main/data/cath_3di
TMP_DIR=/home/zww20/rds/hpc-work/GDL/PiFold-main/data/tmp

mkdir -p $OUT_DIR $TMP_DIR

$FOLDSEEK createdb $PDB_DIR $TMP_DIR/cath_db
$FOLDSEEK lndb $TMP_DIR/cath_db_h $TMP_DIR/cath_db_ss_h
$FOLDSEEK convert2fasta $TMP_DIR/cath_db_ss $OUT_DIR/cath_3di.fasta

echo "Done — 3Di tokens saved to $OUT_DIR/cath_3di.fasta"
