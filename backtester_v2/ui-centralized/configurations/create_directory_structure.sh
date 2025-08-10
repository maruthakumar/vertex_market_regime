#!/bin/bash

# Script to create directory structure in all locations
echo "Creating directory structure for Excel configurations..."

# Define all locations
LOCATIONS=(
    "/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-tv/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-tbs/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-pos/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-oi/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-orb/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-ml-indicator/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-market-regime/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-indicator/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-quantum/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/ml-systems/ml-core-system/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/ml-systems/ml-straddle-system/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/ml-systems/ml-triple-straddle/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/core-refactor/api-enhancement/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/core-refactor/core-components/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/core-refactor/dal-optimization/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/integrations/algobaba-integration/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/integrations/zerodha-integration/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/tools/configurations/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/tools/consolidator-optimizer/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/backtester_v2/configurations"
    "/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-strategy-specific/backtester_v2/configurations"
)

SUBDIRS="prod dev archive"
STRATEGIES="tv tbs pos oi orb ml mr"

# Create directories
for LOC in "${LOCATIONS[@]}"; do
    echo "Processing: $LOC"
    
    # Check if location exists
    if [ ! -d "$LOC" ]; then
        echo "  Warning: Location does not exist: $LOC"
        continue
    fi
    
    # Create data directory
    mkdir -p "$LOC/data"
    
    # Create subdirectories
    for ENV in $SUBDIRS; do
        if [[ "$ENV" == "archive" ]]; then
            mkdir -p "$LOC/data/$ENV"
            echo "  Created: $LOC/data/$ENV"
        else
            for STRAT in $STRATEGIES; do
                mkdir -p "$LOC/data/$ENV/$STRAT"
                echo "  Created: $LOC/data/$ENV/$STRAT"
            done
        fi
    done
done

echo "Directory structure creation completed!"
echo ""
echo "Summary:"
echo "- Created prod/dev/archive structure in all locations"
echo "- Each prod and dev folder contains: tv, tbs, pos, oi, orb, ml, mr subdirectories"
echo "- Archive folder is for old configurations"