#!/bin/bash

# Check if a command line argument is given
if [ $# -eq 1 ]; then
  # Use the command line argument as the commit message
  commit_message="$1"
else
  # Use the date and time as the commit message
  commit_message=$(date +"%Y_%m_%d_%H_%M_commit")
fi

# Add all changes to the index
git add .

# Commit the changes with the given commit message
git commit -m "$commit_message"

# Pull the latest changes from the remote repository
git pull

# Push the changes to the remote repository
git push
