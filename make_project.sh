#!/bin/bash

# This script creates a new project directory and copies the template files
# into it. It also creates a new git repository and commits the template files.

# Usage: make_project.sh <project_name>
PROJECT_NAME = $1

# Check that the project name was given
if [ $# -eq 0 ]; then
    echo "Usage: make_project.sh <project_name>"
    exit 1
fi


# Create the project directory in parent directory
mkdir ../ $PROJECT_NAME

# Copy the template files into the project directory
cp -r ./* ../ $PROJECT_NAME/
cp .gitignore ../ $PROJECT_NAME/
cp .pre-commit-config.yaml ../ $PROJECT_NAME/


# Change to the project directory
cd ../ $PROJECT_NAME/

# Initialize a new git repository
git init

# Add all files to the repository
git add .

# Commit the files
git commit -m "Initial commit"

# Create a new branch for development
git checkout -b develop

# Create a new branch for the first feature
git checkout -b feature/first-feature

# Create a new branch for the first bugfix
git checkout -b bugfix/first-bugfix

# Create a new branch for the first release
git checkout -b release/first-release


# push 
git push --set-upstream origin develop


# checkout to develop
git checkout develop

# Finish
echo "Project created successfully!"
