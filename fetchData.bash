#!/usr/bin/env bash
# Define conversion function
# Convert
function convertFastq () {
  ngm -b -r ref/sacCer3.fa.gz -q seq/"$1".fastq.gz -t 4 -o seq/"$1".bam
  samtools sort seq/"$1".bam -o seq/"$1"_sorted.bam
  samtools mpileup seq/"$1"_sorted.bam | perl -ne 'BEGIN{print "track type=wiggle_0 name=fileName description=fileName\n"};($c, $start, undef, $depth) = split;if ($c ne $lastC) { print "variableStep chrom=$c\n"; };$lastC=$c;next unless $. % 10 ==0;print "$start\t$depth\n" unless $depth<3;' > reed/"$1".wig
  wigToBigWig seq/"$1".wig ref/chromSacCer.sizes seq/"$1".bw
  rm -rf seq/*.bam seq/*.wig seq/*fastq.gz
}


# Create folder structure
if [ ! -d data ]
then
  mkdir data
fi
cd data
if [ ! -d seq ]
then
  mkdir seq
fi
if [ ! -d ref ]
then
  mkdir ref
fi

# Get meta data
# Transcript definitions from Park et al.
wget 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE49026&format=file&file=GSE49026%5FS%2DPAS%2Etxt%2Egz' --output-document='ref/GSE49026_S-PAS.txt'
wget 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE49026&format=file&file=GSE49026%5FS%2DTSS%2Etxt%2Egz' --output-document='ref/GSE49026_S-TSS.txt'

# Get alignment file
wget -4 --output-document='ref/SacCer3.fa.gz' https://hgdownload.soe.ucsc.edu/goldenPath/sacCer3/bigZips/sacCer3.fa.gz
gunzip ref/SacCer3.fa.gz
rm -rf ref/*.gz
wget -4 --output-document='ref/chromSacCer.sizes' https://hgdownload.soe.ucsc.edu/goldenPath/sacCer3/bigZips/sacCer3.chrom.sizes

# Download Mao et al. CPD data
wget 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2109560&format=file&file=GSM2109560%5FUV%5F0hr%5FA2%5Fdipy%5Fbkgd%5Fminus%2Ewig%2Egz' --output-document='seq/0h_A1_minus.wig.gz'
wget 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2109560&format=file&file=GSM2109560%5FUV%5F0hr%5FA2%5Fdipy%5Fbkgd%5Fplus%2Ewig%2Egz' --output-document='seq/0h_A1_plus.wig.gz'
wget 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2109561&format=file&file=GSM2109561%5FUV%5F1hr%5FA3%5Fdipy%5Fbkgd%5Fminus%2Ewig%2Egz' --output-document='seq/1h_A1_minus.wig.gz'
wget 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2109561&format=file&file=GSM2109561%5FUV%5F1hr%5FA3%5Fdipy%5Fbkgd%5Fplus%2Ewig%2Egz' --output-document='seq/1h_A1_plus.wig.gz'
wget 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2109562&format=file&file=GSM2109562%5F0hr%5FUV2%5FA1%5Fdipy%5Fbkgd%5Fminus%2Ewig%2Egz' --output-document='seq/0h_A2_minus.wig.gz'
wget 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2109562&format=file&file=GSM2109562%5F0hr%5FUV2%5FA1%5Fdipy%5Fbkgd%5Fplus%2Ewig%2Egz' --output-document='seq/0h_A2_plus.wig.gz'
wget 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2109563&format=file&file=GSM2109563%5F20min%5FUV2%5FA2%5Fdipy%5Fbkgd%5Fminus%2Ewig%2Egz' --output-document='seq/20m_A2_minus.wig.gz'
wget 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2109563&format=file&file=GSM2109563%5F20min%5FUV2%5FA2%5Fdipy%5Fbkgd%5Fplus%2Ewig%2Egz' --output-document='seq/20m_A2_plus.wig.gz'
wget 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2109564&format=file&file=GSM2109564%5F2hr%5FUV2%5FA3%5Fdipy%5Fbkgd%5Fminus%2Ewig%2Egz' --output-document='seq/2h_A2_minus.wig.gz'
wget 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM2109564&format=file&file=GSM2109564%5F2hr%5FUV2%5FA3%5Fdipy%5Fbkgd%5Fplus%2Ewig%2Egz' --output-document='seq/2h_A2_plus.wig.gz'

wigToBigWig seq/0h_A1_minus.wig.gz ref/chromSacCer.sizes seq/0h_A1_minus.bw
wigToBigWig seq/0h_A1_plus.wig.gz ref/chromSacCer.sizes seq/0h_A1_plus.bw
wigToBigWig seq/1h_A1_minus.wig.gz ref/chromSacCer.sizes seq/1h_A1_minus.bw
wigToBigWig seq/1h_A1_plus.wig.gz ref/chromSacCer.sizes seq/1h_A1_plus.bw
wigToBigWig seq/0h_A2_minus.wig.gz ref/chromSacCer.sizes seq/0h_A2_minus.bw
wigToBigWig seq/0h_A2_plus.wig.gz ref/chromSacCer.sizes seq/0h_A2_plus.bw
wigToBigWig seq/20m_A2_minus.wig.gz ref/chromSacCer.sizes seq/20m_A2_minus.bw
wigToBigWig seq/20m_A2_plus.wig.gz ref/chromSacCer.sizes seq/20m_A2_plus.bw
wigToBigWig seq/2h_A2_minus.wig.gz ref/chromSacCer.sizes seq/2h_A2_minus.bw
wigToBigWig seq/2h_A2_plus.wig.gz ref/chromSacCer.sizes seq/2h_A2_plus.bw
rm -rf seq/*.wig.gz

# Download netseq data from Harlen et al.
wget 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM1673641&format=file&file=GSM1673641%5FWT%5FNETseq%5FYSC001%5FMinus%2Ebedgraph%2Egz' --output-document='seq/wt_netseq_minus.bedgraph.gz'
wget 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM1673641&format=file&file=GSM1673641%5FWT%5FNETseq%5FYSC001%5FPlus%2Ebedgraph%2Egz' --output-document='seq/wt_netseq_plus.bedgraph.gz'

bedGraphToBigWig seq/wt_netseq_minus.bedgraph.gz ref/chromSacCer.sizes seq/wt_netseq_minus.bw
bedGraphToBigWig seq/wt_netseq_plus.bedgraph.gz ref/chromSacCer.sizes seq/wt_netseq_plus.bw
rm -rf seq/*.bedgraph.gz

# Download MNase, Abf1 and gamma H2A X data from van Eijk et al.
wget 'ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR236/002/ERR2367082/ERR2367082.fastq.gz' --output-document='seq/wt_abf1_nouv.fastq.gz'
wget 'ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR236/000/ERR2367080/ERR2367080.fastq.gz' --output-document='seq/wt_abf1_uv.fastq.gz'
wget 'ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR236/002/ERR2367062/ERR2367062.fastq.gz' --output-document='seq/wt_h2a_0m.fastq.gz'
wget 'ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR236/006/ERR2367076/ERR2367076.fastq.gz' --output-document='seq/nucl_wt_nouv.fastq.gz'
wget 'ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR236/004/ERR2367074/ERR2367074.fastq.gz' --output-document='seq/nucl_wt_0min.fastq.gz'
wget 'ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR236/005/ERR2367075/ERR2367075.fastq.gz' --output-document='seq/nucl_wt_30min.fastq.gz'

convertFastq wt_abf1_nouv
convertFastq wt_abf1_uv
convertFastq wt_h2a_0m
convertFastq nucl_wt_nouv
convertFastq nucl_wt_0min
convertFastq nucl_wt_30min

