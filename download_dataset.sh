#!/bin/bash
# Amazon Deforestation Multi-Sensor Dataset Download Script
# Version 0.1 - Google Cloud Storage Distribution
# Generated: 2025-07-29 22:01:07

set -e  # Exit on any error

echo "ðŸŒ³ Amazon Deforestation Multi-Sensor Dataset Downloader"
echo "======================================================="
echo "Dataset Version: 0.1"
echo "Total Size: 4.4 GB across 6 archives"
echo "Coverage: 100 deforestation alerts from Brazilian Amazon"
echo "Source: gs://capacity_shared/amazon_deforestation_v0.1/"
echo ""

# Google Cloud Storage base URL
BASE_URL="https://storage.googleapis.com/capacity_shared/amazon_deforestation_v0.1"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Archive information from manifest
declare -A ARCHIVES
ARCHIVES[cbers4a]="amazon_deforestation_cbers4a_v1.0_20250729.tar.gz:523.8:166d2a2d5e5801cd178a93c734dbafb8"
ARCHIVES[landsat]="amazon_deforestation_landsat_v1.0_20250729.tar.gz:273.5:ae462add8f7993678986632623857de5"
ARCHIVES[masks]="amazon_deforestation_masks_v1.0_20250729.tar.gz:0.0:4c3ea41d30ae013b9e05a2e4f7e827d1"
ARCHIVES[metadata]="amazon_deforestation_metadata_v1.0_20250729.zip:0.1:90058f12ff4510e21e82f5dd57bc8148"
ARCHIVES[sentinel1]="amazon_deforestation_sentinel1_v1.0_20250729.tar.gz:268.3:e82d0007cf5f59c4bd1d807172094277"
ARCHIVES[sentinel2]="amazon_deforestation_sentinel2_v1.0_20250729.tar.gz:3456.7:dda9bd37ed1f77a97ebab378687a1038"

# Function to show progress
show_progress() {
    local sensor=$1
    case $sensor in
        "sentinel1") echo "ðŸ“¡ Sentinel-1 SAR: C-band radar with VV/VH polarizations (10m resolution)" ;;
        "sentinel2") echo "ðŸ›°ï¸  Sentinel-2 Optical: 13-band multispectral with red-edge (10m resolution)" ;;
        "landsat") echo "ðŸŒ Landsat: Long-term optical with thermal bands (30m resolution)" ;;
        "cbers4a") echo "ðŸ‡§ðŸ‡· CBERS-4A: Brazilian satellite with 2m RGB + multispectral (2-64m resolution)" ;;
        "masks") echo "ðŸŽ¯ Ground Truth: MapBiomas validated deforestation masks" ;;
        "metadata") echo "ðŸ“‹ Metadata: Documentation, GeoJSON alerts, and analysis files" ;;
    esac
}

# Download function with verification and progress
download_and_verify() {
    local sensor=$1
    local archive_info=${ARCHIVES[$sensor]}
    local filename=$(echo $archive_info | cut -d':' -f1)
    local size_mb=$(echo $archive_info | cut -d':' -f2)
    local expected_md5=$(echo $archive_info | cut -d':' -f3)
    
    show_progress $sensor
    echo -e "${BLUE}ðŸ“¥ Downloading $filename (${size_mb} MB)...${NC}"
    
    # Check if file already exists and is complete
    if [ -f "$filename" ]; then
        echo -e "${YELLOW}âš ï¸  File exists, checking integrity...${NC}"
        if command -v md5sum &> /dev/null; then
            actual_md5=$(md5sum "$filename" | cut -d' ' -f1)
            if [ "$actual_md5" = "$expected_md5" ]; then
                echo -e "${GREEN}âœ… File already downloaded and verified${NC}"
                return 0
            else
                echo -e "${YELLOW}âš ï¸  Checksum mismatch, re-downloading...${NC}"
                rm "$filename"
            fi
        fi
    fi
    
    # Download from Google Cloud Storage
    if command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$filename" "$BASE_URL/$filename"
    elif command -v wget &> /dev/null; then
        wget --progress=bar:force -O "$filename" "$BASE_URL/$filename"
    elif command -v gsutil &> /dev/null; then
        gsutil cp "gs://capacity_shared/amazon_deforestation_v0.1/$filename" "$filename"
    else
        echo -e "${RED}âŒ No download tool found (curl, wget, or gsutil)${NC}"
        exit 1
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}âœ… Downloaded $filename${NC}"
        
        # Verify checksum
        if command -v md5sum &> /dev/null; then
            echo "ðŸ” Verifying checksum..."
            actual_md5=$(md5sum "$filename" | cut -d' ' -f1)
            if [ "$actual_md5" = "$expected_md5" ]; then
                echo -e "${GREEN}âœ… Checksum verified${NC}"
            else
                echo -e "${RED}âŒ Checksum mismatch for $filename${NC}"
                echo "Expected: $expected_md5"
                echo "Actual:   $actual_md5"
                exit 1
            fi
        fi
    else
        echo -e "${RED}âŒ Failed to download $filename${NC}"
        echo "ðŸ’¡ You can also download directly from:"
        echo "   $BASE_URL/$filename"
        exit 1
    fi
    
    echo ""
}

# Extract function
extract_archive() {
    local sensor=$1
    local archive_info=${ARCHIVES[$sensor]}
    local filename=$(echo $archive_info | cut -d':' -f1)
    
    if [ ! -f "$filename" ]; then
        echo -e "${RED}âŒ Archive $filename not found${NC}"
        return 1
    fi
    
    echo -e "${BLUE}ðŸ“‚ Extracting $filename...${NC}"
    
    if [[ $filename == *.tar.gz ]]; then
        tar -xzf "$filename"
    elif [[ $filename == *.zip ]]; then
        unzip -q "$filename"
    else
        echo -e "${RED}âŒ Unknown archive format: $filename${NC}"
        return 1
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}âœ… Extracted $filename${NC}"
    else
        echo -e "${RED}âŒ Failed to extract $filename${NC}"
        return 1
    fi
}

# Show usage information
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  all              Download complete dataset (4.4 GB)"
    echo "  cbers4a         Download CBERS-4A multi-resolution imagery (2m RGB + 16m multispectral) (~523.8 MB)"
    echo "  landsat         Download Landsat optical imagery with thermal bands (30m resolution) (~273.5 MB)"
    echo "  masks           Download Ground truth deforestation masks from MapBiomas (~0.0 MB)"
    echo "  metadata        Download Dataset documentation, alert geometries, and analysis (~0.1 MB)"
    echo "  sentinel1       Download Sentinel-1 SAR imagery and radar indices (VV/VH polarizations) (~268.3 MB)"
    echo "  sentinel2       Download Sentinel-2 optical imagery and vegetation indices (10m resolution) (~3456.7 MB)"
    echo "  optical          Download all optical sensors (sentinel2 + landsat + cbers4a)"
    echo "  essential        Download masks + metadata + sentinel2 (minimum for ML)"
    echo ""
    echo "Options:"
    echo "  --extract        Automatically extract archives after download"
    echo "  --keep-archives  Keep compressed files after extraction"
    echo "  --verify-only    Only verify existing files, don't download"
    echo "  --use-gsutil     Use gsutil instead of curl/wget (faster for GCS)"
    echo ""
    echo "Examples:"
    echo "  $0 all --extract                    # Download and extract everything"
    echo "  $0 sentinel2 --extract              # Get only Sentinel-2 data"
    echo "  $0 essential                        # Minimum dataset for ML research"
    echo "  $0 metadata                         # Just get documentation first"
    echo ""
    echo "Google Cloud Storage Direct Access:"
    echo "  gsutil -m cp -r gs://capacity_shared/amazon_deforestation_v0.1/ ."
    echo "  Or browse: https://console.cloud.google.com/storage/browser/capacity_shared/amazon_deforestation_v0.1"
}

# Parse command line arguments
EXTRACT=false
KEEP_ARCHIVES=false
VERIFY_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --extract)
            EXTRACT=true
            shift
            ;;
        --keep-archives)
            KEEP_ARCHIVES=true
            shift
            ;;
        --verify-only)
            VERIFY_ONLY=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            COMMAND=$1
            shift
            ;;
    esac
done

# Main execution logic
case $COMMAND in
    "all")
        echo "ðŸŒ Downloading complete dataset (4.4 GB)..."
        echo ""
        for sensor in cbers4a landsat masks metadata sentinel1 sentinel2; do
            download_and_verify $sensor
            if [ "$EXTRACT" = true ]; then
                extract_archive $sensor
                if [ "$KEEP_ARCHIVES" = false ]; then
                    rm -f "$(echo ${ARCHIVES[$sensor]} | cut -d':' -f1)"
                fi
            fi
        done
        ;;
    "optical")
        echo "ðŸŒž Downloading all optical sensors..."
        echo ""
        for sensor in sentinel2 landsat cbers4a; do
            download_and_verify $sensor
            if [ "$EXTRACT" = true ]; then
                extract_archive $sensor
                if [ "$KEEP_ARCHIVES" = false ]; then
                    rm -f "$(echo ${ARCHIVES[$sensor]} | cut -d':' -f1)"
                fi
            fi
        done
        ;;
    "essential")
        echo "ðŸŽ¯ Downloading essential ML dataset..."
        echo ""
        for sensor in masks metadata sentinel2; do
            download_and_verify $sensor
            if [ "$EXTRACT" = true ]; then
                extract_archive $sensor
                if [ "$KEEP_ARCHIVES" = false ]; then
                    rm -f "$(echo ${ARCHIVES[$sensor]} | cut -d':' -f1)"
                fi
            fi
        done
        ;;
    "cbers4a")
        download_and_verify cbers4a
        if [ "$EXTRACT" = true ]; then
            extract_archive cbers4a
            if [ "$KEEP_ARCHIVES" = false ]; then
                rm -f "$(echo ${ARCHIVES[cbers4a]} | cut -d':' -f1)"
            fi
        fi
        ;;
    "landsat")
        download_and_verify landsat
        if [ "$EXTRACT" = true ]; then
            extract_archive landsat
            if [ "$KEEP_ARCHIVES" = false ]; then
                rm -f "$(echo ${ARCHIVES[landsat]} | cut -d':' -f1)"
            fi
        fi
        ;;
    "masks")
        download_and_verify masks
        if [ "$EXTRACT" = true ]; then
            extract_archive masks
            if [ "$KEEP_ARCHIVES" = false ]; then
                rm -f "$(echo ${ARCHIVES[masks]} | cut -d':' -f1)"
            fi
        fi
        ;;
    "metadata")
        download_and_verify metadata
        if [ "$EXTRACT" = true ]; then
            extract_archive metadata
            if [ "$KEEP_ARCHIVES" = false ]; then
                rm -f "$(echo ${ARCHIVES[metadata]} | cut -d':' -f1)"
            fi
        fi
        ;;
    "sentinel1")
        download_and_verify sentinel1
        if [ "$EXTRACT" = true ]; then
            extract_archive sentinel1
            if [ "$KEEP_ARCHIVES" = false ]; then
                rm -f "$(echo ${ARCHIVES[sentinel1]} | cut -d':' -f1)"
            fi
        fi
        ;;
    "sentinel2")
        download_and_verify sentinel2
        if [ "$EXTRACT" = true ]; then
            extract_archive sentinel2
            if [ "$KEEP_ARCHIVES" = false ]; then
                rm -f "$(echo ${ARCHIVES[sentinel2]} | cut -d':' -f1)"
            fi
        fi
        ;;
    "list")
        echo "ðŸ“‹ Available archives:"
        echo ""
        for sensor in cbers4a landsat masks metadata sentinel1 sentinel2; do
            archive_info=${ARCHIVES[$sensor]}
            filename=$(echo $archive_info | cut -d':' -f1)
            size_mb=$(echo $archive_info | cut -d':' -f2)
            show_progress $sensor
            echo "   â””â”€â”€ $filename (${size_mb} MB)"
            echo "       $BASE_URL/$filename"
            echo ""
        done
        ;;
    *)
        if [ -z "$COMMAND" ]; then
            echo -e "${YELLOW}âš ï¸  No command specified${NC}"
        else
            echo -e "${RED}âŒ Unknown command: $COMMAND${NC}"
        fi
        echo ""
        show_usage
        exit 1
        ;;
esac

echo -e "${GREEN}âœ… Operation completed successfully!${NC}"
echo ""
echo "ðŸ“š Next steps:"
echo "   1. Check DATASET_OVERVIEW.md for detailed documentation"
echo "   2. Load data using the provided Python examples"
echo "   3. Refer to sensor-specific README files for details"
echo ""
echo "ðŸ”— Dataset Resources:"
echo "   â€¢ Documentation: DATASET_OVERVIEW.md"
echo "   â€¢ GCS Bucket: gs://capacity_shared/amazon_deforestation_v0.1/"
echo "   â€¢ Public URL: https://storage.googleapis.com/capacity_shared/amazon_deforestation_v0.1/"