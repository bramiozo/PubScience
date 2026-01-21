#!/bin/bash

# Bash script to run GCS file transfer in detached mode
# Usage: ./run_gcs_transfer.sh <source_dir> <gcs_bucket> [prefix] [options]

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    echo "Usage: $0 <source_dir> <gcs_bucket> [prefix] [options]"
    echo ""
    echo "Arguments:"
    echo "  source_dir    Local directory containing files to upload"
    echo "  gcs_bucket    GCS bucket name (e.g., 'my-bucket' or 'gs://my-bucket')"
    echo "  prefix        Optional: GCS prefix/folder path (default: '')"
    echo ""
    echo "Options:"
    echo "  --dry-run     Perform a dry run without uploading"
    echo "  --workers N   Number of parallel workers (default: 10)"
    echo "  --no-detach   Run in foreground instead of detached mode"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/files my-bucket"
    echo "  $0 /path/to/files my-bucket data/2024"
    echo "  $0 /path/to/files my-bucket data/2024 --dry-run"
    echo "  $0 /path/to/files my-bucket --workers 20"
    exit 1
}

# Check if help is requested
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]] || [[ $# -lt 2 ]]; then
    usage
fi

# Parse arguments
SOURCE_DIR="$1"
GCS_BUCKET="$2"
PREFIX="${3:-}"
DETACH_MODE=true
DRY_RUN=""
WORKERS="10"

# Shift positional parameters if prefix is actually an option
if [[ "$PREFIX" == --* ]]; then
    PREFIX=""
    shift 2
else
    shift 3
fi

# Parse remaining options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --no-detach)
            DETACH_MODE=false
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate source directory
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo -e "${RED}Error: Source directory '$SOURCE_DIR' does not exist${NC}"
    exit 1
fi

# Get absolute path
SOURCE_DIR=$(realpath "$SOURCE_DIR")

# Remove gs:// prefix if present
GCS_BUCKET="${GCS_BUCKET#gs://}"

# Set up directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$SCRIPT_DIR/.gcs_transfer.pid"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/gcs_transfer_${TIMESTAMP}.log"
ERROR_FILE="$LOG_DIR/gcs_transfer_${TIMESTAMP}.err"

# Check if Python script exists
PYTHON_SCRIPT="$SCRIPT_DIR/move_to_gcs.py"
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo -e "${RED}Error: Python script not found at $PYTHON_SCRIPT${NC}"
    exit 1
fi

# Check if another instance is running
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Another transfer process is already running (PID: $OLD_PID)${NC}"
        echo "Would you like to:"
        echo "  1) Kill the existing process and start a new one"
        echo "  2) Exit without starting a new transfer"
        read -p "Choose (1/2): " choice

        case $choice in
            1)
                echo "Killing existing process..."
                kill "$OLD_PID"
                sleep 2
                if ps -p "$OLD_PID" > /dev/null 2>&1; then
                    kill -9 "$OLD_PID"
                fi
                rm -f "$PID_FILE"
                ;;
            2)
                echo "Exiting..."
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice. Exiting...${NC}"
                exit 1
                ;;
        esac
    else
        rm -f "$PID_FILE"
    fi
fi

# Build the command
CMD="python3 $PYTHON_SCRIPT"
CMD="$CMD --source-dir '$SOURCE_DIR'"
CMD="$CMD --bucket '$GCS_BUCKET'"
[[ -n "$PREFIX" ]] && CMD="$CMD --prefix '$PREFIX'"
[[ -n "$DRY_RUN" ]] && CMD="$CMD --dry-run"
CMD="$CMD --workers $WORKERS"

# Display transfer information
echo -e "${GREEN}GCS File Transfer Configuration:${NC}"
echo "  Source Directory: $SOURCE_DIR"
echo "  GCS Bucket: gs://$GCS_BUCKET/$PREFIX"
echo "  Workers: $WORKERS"
[[ -n "$DRY_RUN" ]] && echo "  Mode: DRY RUN"
echo "  Log file: $LOG_FILE"
echo ""

# Check for Google Cloud credentials
if [[ -z "${GOOGLE_APPLICATION_CREDENTIALS}" ]] && ! command -v gcloud &> /dev/null; then
    echo -e "${YELLOW}Warning: GOOGLE_APPLICATION_CREDENTIALS not set and gcloud not found.${NC}"
    echo "Make sure you have proper authentication set up for Google Cloud Storage."
    echo ""
fi

# Run the transfer
if [[ "$DETACH_MODE" == true ]]; then
    echo -e "${GREEN}Starting transfer in detached mode...${NC}"

    # Start the process in background with nohup
    nohup bash -c "
        echo \$\$ > '$PID_FILE'
        exec $CMD
    " > "$LOG_FILE" 2> "$ERROR_FILE" &

    NEW_PID=$!
    sleep 2

    # Check if process started successfully
    if ps -p "$NEW_PID" > /dev/null 2>&1; then
        # Get the actual Python process PID
        ACTUAL_PID=$(cat "$PID_FILE" 2>/dev/null || echo "$NEW_PID")
        echo -e "${GREEN}Transfer started successfully (PID: $ACTUAL_PID)${NC}"
        echo ""
        echo "Monitor progress with:"
        echo "  tail -f $LOG_FILE"
        echo ""
        echo "Check status with:"
        echo "  ps -p $ACTUAL_PID"
        echo ""
        echo "Stop transfer with:"
        echo "  kill $ACTUAL_PID"
    else
        echo -e "${RED}Failed to start transfer process${NC}"
        echo "Check error log: $ERROR_FILE"
        exit 1
    fi
else
    echo -e "${GREEN}Starting transfer in foreground mode...${NC}"
    echo ""
    eval "$CMD" 2>&1 | tee "$LOG_FILE"
fi
