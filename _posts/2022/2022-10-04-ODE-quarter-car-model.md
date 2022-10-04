---
layout: post
title:  "ODE: quarted car model solver with scipy"
date:   2022-10-04 20:00:00 +0100
categories: data-analysis
---

Here we show how one can use scipy to solve a simple differential equation. We use the so called quarter car model as an example. This simple model describes the motion of a mass connected to a deformable wheel through a dumper-spring system. 



<p style="text-align:center;"><img src="/asset/images/scipy/quarter_car_eq.png" alt="quarter_car_equation" higth="300"></p>


## Code
Here is the code to find the maximum Chow coefficient in a given dataset. Note that we can change the `cuts_num` argument to increase and decrease the number of cuts in the dataset. The function `find_max_chow` also returns information about the linear fits corresponding to the cut location of the max Chow coefficient found.

<p style="text-align:center;"><img src="/asset/images/data-exploration/chow_static.png" alt="chow example" width="400"></p>
