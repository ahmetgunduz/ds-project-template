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

sed -i "s/name: catchjoe/name: $PROJECT_NAME/g" ./conda.yaml

# Initialize a new git repository
git init

# Add all files to the repository
git add .

# Commit the files
git commit -m "Initial commit"

# Create a new branch for development
git checkout -b develop





echo "To start working on the project, run the following commands:"
echo "git checkout develop"

cd ../$PROJECT_NAME/

echo "Changed to the project directory"


# Finish
echo "Project created successfully! Happy coding!"