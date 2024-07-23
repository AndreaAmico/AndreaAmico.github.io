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


### Extra github trick
Quickely edit the repository using VSCode online by changing **github.com** to **github.dev** in the repository link.