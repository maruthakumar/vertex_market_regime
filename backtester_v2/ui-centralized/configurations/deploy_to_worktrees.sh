#!/bin/bash

# Script to deploy configuration files to relevant worktrees
echo "Deploying configuration files to worktrees..."

# Source directory
SOURCE_DIR="/srv/samba/shared/bt/backtester_stable/BTRUN/backtester_v2/configurations/data/prod"

# Function to copy strategy files
copy_strategy_files() {
    local strategy=$1
    local target_dir=$2
    
    if [ -d "$SOURCE_DIR/$strategy" ]; then
        echo "  Copying $strategy files to $target_dir"
        mkdir -p "$target_dir/$strategy"
        cp -r "$SOURCE_DIR/$strategy/"*.xlsx "$target_dir/$strategy/" 2>/dev/null
    fi
}

# Deploy to strategy-specific worktrees
echo "Deploying to strategy-specific worktrees..."

# TV Strategy
TARGET="/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-tv/backtester_v2/configurations/data/prod"
mkdir -p "$TARGET"
copy_strategy_files "tv" "$TARGET"

# TBS Strategy
TARGET="/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-tbs/backtester_v2/configurations/data/prod"
mkdir -p "$TARGET"
copy_strategy_files "tbs" "$TARGET"

# POS Strategy
TARGET="/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-pos/backtester_v2/configurations/data/prod"
mkdir -p "$TARGET"
copy_strategy_files "pos" "$TARGET"

# OI Strategy
TARGET="/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-oi/backtester_v2/configurations/data/prod"
mkdir -p "$TARGET"
copy_strategy_files "oi" "$TARGET"

# ORB Strategy
TARGET="/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-orb/backtester_v2/configurations/data/prod"
mkdir -p "$TARGET"
copy_strategy_files "orb" "$TARGET"

# ML Indicator Strategy
TARGET="/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-ml-indicator/backtester_v2/configurations/data/prod"
mkdir -p "$TARGET"
copy_strategy_files "ml" "$TARGET"

# Market Regime Strategy
TARGET="/srv/samba/shared/bt/backtester_stable/worktrees/strategies/strategy-market-regime/backtester_v2/configurations/data/prod"
mkdir -p "$TARGET"
copy_strategy_files "mr" "$TARGET"

# Deploy ML files to ML system worktrees
echo "Deploying ML files to ML system worktrees..."
for WORKTREE in "ml-core-system" "ml-straddle-system" "ml-triple-straddle"; do
    TARGET="/srv/samba/shared/bt/backtester_stable/worktrees/ml-systems/$WORKTREE/backtester_v2/configurations/data/prod"
    mkdir -p "$TARGET"
    copy_strategy_files "ml" "$TARGET"
done

# Deploy all strategy files to integration worktrees (they might need all strategies)
echo "Deploying all strategies to integration worktrees..."
for WORKTREE in "algobaba-integration" "zerodha-integration"; do
    TARGET="/srv/samba/shared/bt/backtester_stable/worktrees/integrations/$WORKTREE/backtester_v2/configurations/data/prod"
    mkdir -p "$TARGET"
    for STRATEGY in tv tbs pos oi orb ml mr; do
        copy_strategy_files "$STRATEGY" "$TARGET"
    done
done

# Deploy all strategies to core refactor worktrees
echo "Deploying all strategies to core refactor worktrees..."
for WORKTREE in "api-enhancement" "core-components" "dal-optimization"; do
    TARGET="/srv/samba/shared/bt/backtester_stable/worktrees/core-refactor/$WORKTREE/backtester_v2/configurations/data/prod"
    mkdir -p "$TARGET"
    for STRATEGY in tv tbs pos oi orb ml mr; do
        copy_strategy_files "$STRATEGY" "$TARGET"
    done
done

# Deploy all strategies to tools worktrees
echo "Deploying all strategies to tools worktrees..."
for WORKTREE in "configurations" "consolidator-optimizer"; do
    TARGET="/srv/samba/shared/bt/backtester_stable/worktrees/tools/$WORKTREE/backtester_v2/configurations/data/prod"
    mkdir -p "$TARGET"
    for STRATEGY in tv tbs pos oi orb ml mr; do
        copy_strategy_files "$STRATEGY" "$TARGET"
    done
done

# Deploy all strategies to UI refactor worktrees
echo "Deploying all strategies to UI refactor worktrees..."
for WORKTREE in "ui-centralized" "ui-strategy-specific"; do
    TARGET="/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/$WORKTREE/backtester_v2/configurations/data/prod"
    mkdir -p "$TARGET"
    for STRATEGY in tv tbs pos oi orb ml mr; do
        copy_strategy_files "$STRATEGY" "$TARGET"
    done
done

echo "Deployment completed!"

# Summary
echo ""
echo "Summary of deployment:"
echo "- Strategy-specific worktrees: Received their respective strategy files"
echo "- ML system worktrees: Received ML strategy files"
echo "- Integration worktrees: Received all strategy files"
echo "- Core refactor worktrees: Received all strategy files"
echo "- Tools worktrees: Received all strategy files"
echo "- UI refactor worktrees: Received all strategy files"