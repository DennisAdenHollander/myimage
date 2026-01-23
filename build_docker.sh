###############################################################
################### REMOVE WHEN DELIVERING ####################
###############################################################
set -e #stop when error

# 1. Clean up stopped containers
#docker container prune -f

# 2. Clean up dangling images (no tag, old intermediates)
#docker image prune -f

# 3. Clean up old build cache
#docker builder prune -f
###############################################################

docker build --no-cache --pull --tag "dennis:latest" .
