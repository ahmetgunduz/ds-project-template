#!/bin/bash

# This script creates a new project directory and copies the template files
# into it. It also creates a new git repository and commits the template files.

# Usage: make_project.sh <project_name>
# read the project name from the command line
PROJECT_NAME=$1

# Check that the project name was given
if [ $# -eq 0 ]; then
    echo "Usage: make_project.sh <project_name>"
    exit 1
fi


# Create the project directory in parent directory
# check if there is a directory with the same name
if [ -d ../$PROJECT_NAME ]; then
    echo "Directory already exists"
    exit 1
fi

mkdir ../$PROJECT_NAME

# Copy the template files into the project directory
cp -r ./* ../$PROJECT_NAME/
cp .gitignore ../$PROJECT_NAME/
cp .pre-commit-config.yaml ../$PROJECT_NAME/


# Change to the project directory
cd ../$PROJECT_NAME/

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




echo "To start working on the project, run the following commands:"
echo "git checkout develop"
echo "git checkout -b feature/<feature_name>"
echo "git checkout -b bugfix/<bugfix_name>"
echo "git checkout -b release/<release_name>"
echo "git checkout -b hotfix/<hotfix_name>"
echo "git checkout -b support/<support_name>"
echo "git checkout -b documentation/<documentation_name>"
echo "git checkout -b <other_name>"

cd ../$PROJECT_NAME/

echo "Changed to the project directory"


# Finish
echo "Project created successfully! Happy coding!"