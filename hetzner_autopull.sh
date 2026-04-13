#!/bin/bash
# =============================================================================
# SentinelAI -- Hetzner Auto-Pull (Cronjob)
# Keeps the Hetzner server in sync with GitHub/main.
#
# Cronjob setup (run: crontab -e):
#   Every 15 minutes:
#   */15 * * * * /home/cap/hetzner_autopull.sh >> /home/cap/logs/autopull.log 2>&1
#
#   Once daily at 06:00:
#   0 6 * * * /home/cap/hetzner_autopull.sh >> /home/cap/logs/autopull.log 2>&1
# =============================================================================

REPO_PATH="/home/cap/vigilex"
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')]"

echo "$LOG_PREFIX ===== autopull start ====="

# Check if repo folder exists and is a git repo
if [ ! -d "$REPO_PATH/.git" ]; then
    echo "$LOG_PREFIX ERROR: $REPO_PATH is not a git repo"
    exit 1
fi

cd "$REPO_PATH"

# Log current branch and commit
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
CURRENT_COMMIT=$(git rev-parse --short HEAD)
echo "$LOG_PREFIX Branch: $CURRENT_BRANCH | Commit before pull: $CURRENT_COMMIT"

# Fetch without merging
git fetch origin

# Check if there are new commits
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "$LOG_PREFIX Already up to date -- no pull needed."
    echo "$LOG_PREFIX ===== autopull end ====="
    exit 0
fi

# New commits found -- pull
echo "$LOG_PREFIX New commits found, pulling..."
git pull origin main

if [ $? -eq 0 ]; then
    NEW_COMMIT=$(git rev-parse --short HEAD)
    echo "$LOG_PREFIX Pull successful: $CURRENT_COMMIT -> $NEW_COMMIT"

    # Optional: restart Docker services after pull
    # Uncomment once docker-compose is running (Phase 2)
    # echo "$LOG_PREFIX Restarting Docker services..."
    # docker compose -f "$REPO_PATH/docker-compose.yml" up -d --build
    # echo "$LOG_PREFIX Docker restart: exit $?"
else
    echo "$LOG_PREFIX ERROR: git pull failed"
    exit 1
fi

echo "$LOG_PREFIX ===== autopull end ====="
