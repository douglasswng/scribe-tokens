#!/bin/bash
# Train all task-repr combinations in parallel batches
# Trains K models at a time, iterating over task then repr

# Configuration
K=4  # Number of models to train in parallel (change this value as needed)
TASKS=("HTR" "HTG" "NTP" "HTR_SFT" "HTG_SFT")  # Excluding HTG_GRPO for now
REPRS=("scribe" "point5" "rel" "text")  # Excluding point3 and abs for now
PYTHON_CMD=".venv/bin/python"
SCRIPT_PATH="scripts.train.train"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Training Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to check if a tmux session exists
tmux_session_exists() {
    tmux has-session -t "$1" 2>/dev/null
}

# Function to wait for a list of tmux sessions to complete
wait_for_sessions() {
    local -n sessions=$1  # Use nameref to accept array by reference

    if [ ${#sessions[@]} -eq 0 ]; then
        return
    fi

    echo -e "${YELLOW}Waiting for batch of ${#sessions[@]} training sessions to complete...${NC}"

    while true; do
        local all_done=true

        for session_name in "${sessions[@]}"; do
            if tmux_session_exists "$session_name"; then
                all_done=false
                break
            fi
        done

        if [ "$all_done" = true ]; then
            echo -e "${GREEN}✓ All training sessions in batch completed${NC}"
            break
        fi

        # Show status
        local running_count=0
        for session_name in "${sessions[@]}"; do
            if tmux_session_exists "$session_name"; then
                ((running_count++))
            fi
        done

        echo -e "${BLUE}[$(date '+%H:%M:%S')] ${running_count}/${#sessions[@]} sessions still running...${NC}"
        sleep 10  # Check every 10 seconds
    done
}

# Generate all task-repr combinations in order (task then repr)
combinations=()
for task in "${TASKS[@]}"; do
    for repr in "${REPRS[@]}"; do
        combinations+=("${task}:${repr}")
    done
done

# Main training loop - process K models at a time
total_models=${#combinations[@]}
batch_num=1

echo -e "${BLUE}Total models to train: ${total_models}${NC}"
echo -e "${BLUE}Parallel training limit (K): ${K}${NC}"
echo ""

for ((i=0; i<total_models; i+=K)); do
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Batch ${batch_num}: Models $((i+1))-$((i+K > total_models ? total_models : i+K)) of ${total_models}${NC}"
    echo -e "${BLUE}========================================${NC}"

    # Start K models in parallel
    batch_sessions=()
    for ((j=0; j<K && i+j<total_models; j++)); do
        combination="${combinations[$((i+j))]}"
        task="${combination%%:*}"
        repr="${combination#*:}"
        session_name="${task}_${repr}"

        # Kill existing tmux session if it exists
        if tmux_session_exists "$session_name"; then
            echo -e "${YELLOW}⚠ Tmux session '$session_name' already exists, killing it...${NC}"
            tmux kill-session -t "$session_name"
            sleep 1  # Give it a moment to clean up
        fi

        echo -e "${GREEN}Starting training: Task=${task}, Repr=${repr}${NC}"
        echo -e "${BLUE}  Tmux session: ${session_name}${NC}"

        # Create a detached tmux session and run the training command
        tmux new-session -d -s "$session_name" bash -c "
            cd /home/ubuntu/projects/scribe-tokens
            echo 'Starting training for ${task} - ${repr}'
            echo 'Command: ${PYTHON_CMD} -m ${SCRIPT_PATH} --task ${task} --repr ${repr}'
            ${PYTHON_CMD} -m ${SCRIPT_PATH} --task ${task} --repr ${repr}
            exit_code=\$?
            if [ \$exit_code -eq 0 ]; then
                echo '✓ Training completed successfully for ${task} - ${repr}'
            else
                echo '✗ Training failed for ${task} - ${repr} with exit code \$exit_code'
            fi
        "

        batch_sessions+=("$session_name")
        sleep 1  # Brief delay between starting sessions
    done

    echo ""
    echo -e "${YELLOW}All training sessions for batch ${batch_num} have been started${NC}"
    echo -e "${YELLOW}Active tmux sessions in this batch:${NC}"
    for session_name in "${batch_sessions[@]}"; do
        echo "  - ${session_name}"
    done

    # Wait for all models in this batch to complete
    wait_for_sessions batch_sessions

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Batch ${batch_num} completed!${NC}"
    echo -e "${GREEN}========================================${NC}"

    ((batch_num++))
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All Training Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Summary:${NC}"
echo -e "  Tasks trained: ${TASKS[*]}"
echo -e "  Representations: ${REPRS[*]}"
echo -e "  Total models: $((${#TASKS[@]} * ${#REPRS[@]}))"
