---
layout: post
title:  "Chow: find discontinuity in linear data"
date:   2022-04-18 20:00:00 +0100
categories: data-analysis
---

A simple method to test the presence and locate a discontinuity in a linear trend is to exploit the Chow statistics [see wiki](https://en.wikipedia.org/wiki/Chow_test). The idea is the following: we split the dataset in two chunks by cutting it along the x axis (multiple times). We separately perform linear fits on the two subsets and we compare the residuals to the residuals obtain via a single linear fit on the full dataset, obtaining the Chow coefficient. As we can see in the following animation, if there is a discontinuity on data, it will be highlighted by a higher Chow coefficient. 

<p style="text-align:center;"><img src="/asset/images/data-exploration/chow_animation.gif" alt="chow animation" width="800"></p>
