@echo off
set /p CommitMessage= Commit message: 
git add --all
git commit -m "%CommitMessage%"
git push origin master