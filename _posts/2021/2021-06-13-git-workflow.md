---
layout: post
title:  "Git workflow"
date:   2021-06-13 20:00:00 +0100
categories: other
---



### Clone an existing repo
```bash
git clone ##MY REPO##
```

### Create a new branch
```bash
git checkout master # we want the branch starting from master
git branch feature
git checkout feature
```

### Push a commit in the new branch
```bash
 git add --all # or better, the new files one by one
 git commit
 git push origin feature
 ```



### How to fix a push in a wrong branch:

Identify commit sha (-3 to see last 3 commits):
```bash
git log -1
 ```

Copy COMMIT_SHA.

Checkout to the wrong_branch you just pushed (probably already there):
```bash
git checkout wrong_branch
 ```
Revert the commit (repeat for any other commit sha you want to revert - starting from the most recent):
```bash
git revert COMMIT_SHA
 ```
Push remote if needed (git push -f).

Checkout to the good branch:
```bash
git checkout good_branch
 ```
Cherry pick the commit you want to recover (repeat for any other commit sha you want to revert - starting from the most recent):
```bash
git cherry-pick COMMIT_SHA
 ```


### Extra github trick
Quickely edit the repository using VSCode online by changing **github.com** to **github.dev** in the repository link.