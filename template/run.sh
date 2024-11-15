#!/bin/bash

#SBATCH --job-name="Rosie FastAPI Template - Slurm Job"
#SBATCH --output=output/job_%j.out
#SBATCH --mail-type=ALL
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=06:00:00

## Ensure output and image directories exist
mkdir -p output
mkdir -p image

## Default values
MAIL_USER="student@msoe.edu"
DOCKER_IMAGE="nvidia/cuda:12.5.0-devel-ubuntu22.04"
IMAGE_PATH="./image/container.sif"
FORCE_REBUILD=false

if [ $# -lt 1 ]; then
    echo "Usage: sbatch $0 <required_argument> [-m mail_user] [-i docker_image] [-f (force rebuild)]" >&2
    exit 1
fi

## First argument is the required argument
PASSWORD="$1"
shift # Shift arguments to ignore the first one for getopts

## Create hashed+salted password
export SALT=$(openssl rand -hex 16)
export API_TOKEN=$(echo -n "${SALT}${PASSWORD}" | openssl dgst -sha256 | awk '{print $2}')

## Parse optional command-line arguments
while getopts "m:i:f" opt; do
    case $opt in
        m) MAIL_USER="$OPTARG" ;;
        i) DOCKER_IMAGE="$OPTARG" ;;
        f) FORCE_REBUILD=true ;;
        *) echo "Usage: sbatch $0 <required_argument> [-m mail_user] [-i docker_image] [-f (force rebuild)]" >&2
           exit 1 ;;
    esac
done

## Set the mail user dynamically based on the parsed argument
#SBATCH --mail-user=${MAIL_USER}

## Function to find an open port, starting with a random user port (1024-49151)
find_port() {
    local start_port=1024
    local end_port=49151
    local random_port=$((RANDOM % (end_port - start_port + 1) + start_port))

    # Check the random port first
    if ! (echo > "/dev/tcp/localhost/$random_port") &>/dev/null; then
        echo "$random_port"
        return 0
    fi

    # If the random port is not available, check sequentially from start_port to end_port
    for ((port=start_port; port<=end_port; port++)); do
        if ! (echo > "/dev/tcp/localhost/$port") &>/dev/null; then
            echo "$port"
            return 0
        fi
    done

    echo "No open port found in the range $start_port-$end_port" >&2
    return 1
}

## Function to pull Docker image and build Singularity image if it does not exist or if forced
build_singularity_image() {
    if [[ "$FORCE_REBUILD" == true ]]; then
        echo "Force rebuild enabled. Removing existing Singularity image: $IMAGE_PATH..."
        rm -f "$IMAGE_PATH"
    fi

    if [[ ! -f "$IMAGE_PATH" ]]; then
        echo "Building Singularity image from Docker image: $DOCKER_IMAGE (Force Rebuild: $FORCE_REBUILD)..."
        singularity build "$IMAGE_PATH" "docker://$DOCKER_IMAGE"
        if [[ $? -eq 0 ]]; then
            echo "Singularity image built successfully: $IMAGE_PATH"
        else
            echo "Failed to build Singularity image. Exiting."
            exit 1
        fi
    else
        echo "Singularity image already exists: $IMAGE_PATH"
    fi
}

## SCRIPT START

echo "Executing node hostname: $(hostname)"

# Build the Singularity image if needed
build_singularity_image

# Find an open port and store it in a variable
PORT=$(find_port)
if [ $? -eq 0 ]; then
    echo "Found open port: $PORT"
else
    echo "Failed to find an open port. Exiting."
    exit 1
fi

# Determine the host and URLs
HOST=$(hostname)
export BASE_URL="/node/${HOST}.hpc.msoe.edu/${PORT}"
MODIFIED_URL=$(echo "$BASE_URL" | sed -e 's#^/node/##' -e 's#/[^/]*$##')

echo "BASE_URL: $BASE_URL"
echo "MODIFIED_URL: $MODIFIED_URL"

# Run the Singularity container with the specified command
echo "Running Singularity container..."
singularity exec --nv --network-args portmap=$PORT:$PORT -B /data:/data $IMAGE_PATH uvicorn --app-dir /var/task/app main:app --port $PORT --host $MODIFIED_URL

echo "Singularity container execution finished."

## SCRIPT END
