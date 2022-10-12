---
layout: post
title:  "ODE: quarted car model solver with scipy"
date:   2022-10-04 20:00:00 +0100
categories: other
---

<p style="text-align:center;"><img src="/asset/images/scipy/qc.gif" alt="quarter_car_animation" width="1000"></p>

Here we show how one can use scipy to solve a simple differential equation. We use the so called quarter car model as an example. This simple model describes the motion of a mass connected to a deformable wheel through a dumper-spring system. As shown in the scheme below, the components of the system are the following:
- Suspended mass: mass = `ms`, and hight = `zs`
- Tire: mass = `m_u` (unsuspended), hight = `zu`, and elestic coefficient = `kt`
- Dumper-spring system: dumping coefficient = `cd`, and elestic coefficient = `kd`

<p style="text-align:center;"><img src="/asset/images/scipy/quarter_car_scheme.png" alt="quarter_car_scheme" height="300"></p>

The equations of motions to solve are the following:

<p style="text-align:center;"><img src="/asset/images/scipy/quarter_car_eq.png" alt="quarter_car_equation" height="100"></p>


In python we need to create a function witch takes as inputs the time (`t`)

```python
def quarter_car(t, y, ms, mu, kd, kt, cd, road_function):
    zs, vs, zu, vu = y
    zp = road_function(t)
    
    d1_zs = vs
    d2_zs = (kd*(zu-zs) + cd*(vu-vs))/ms

    d1_zu = vu
    d2_zu = (-kd*(zu-zs)-cd*(vu-vs)+kt*(zp-zu))/mu

    dydt = [d1_zs, d2_zs, d1_zu, d2_zu]
    return dydt
```


## Code
First we need to import the following libraries. Gaussian filter and interpolation will be used to generate a random road profile.
```python
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter
```

We create a random road profile:

```python
profile_length = 150 #m
resolution = 0.005 #m
speed = 80 #Km/h
km_h_to_m_s = 1/3.6
amplitude = 0.5
sigma_gaussian_filter = 500

np.random.seed(41)
profile = pd.DataFrame(dict(
    pc = np.arange(0, profile_length, resolution),
    h = gaussian_filter(np.random.uniform(-amplitude, amplitude, size=int(profile_length/resolution)), sigma_gaussian_filter)
  ))
profile['t'] = profile['pc']/(speed*km_h_to_m_s)
profile.plot(x='t', y='h', figsize=(15,3));
```
<p style="text-align:center;"><img src="/asset/images/scipy/road_profile.png" alt="road_profile" width="800"></p>


We define the road profile function and the equations of motion:
```python
def road_function(t):
    return interp1d(profile.t.values, profile.h.values)(t)

    
def quarter_car(t, y, ms, mu, kd, kt, cd, road_function):
    zs, vs, zu, vu = y
    zp = road_function(t)
    
    d1_zs = vs
    d2_zs = (kd*(zu-zs) + cd*(vu-vs))/ms

    d1_zu = vu
    d2_zu = (-kd*(zu-zs)-cd*(vu-vs)+kt*(zp-zu))/mu

    dydt = [d1_zs, d2_zs, d1_zu, d2_zu]
    return dydt
```

We finally use scipy to solve the equations and plot the results:
```python
y0 = [road_function(0),0,road_function(0),0]
dt = 0.01

tmin = profile.t.min()
tmax = profile.t.max()
xmin = profile.pc.min()
xmax = profile.pc.max()

t = np.arange(tmin, tmax, dt)

kd = 20000
kt = 200000
cd = 2000
ms = 200
mu = 50

sol = solve_ivp(quarter_car, (tmin, tmax), y0, args=(ms, mu, kd, kt, cd, road_function), t_eval=t)

fig, ax = plt.subplots(1, figsize=(15, 3))
plt.plot(t, sol.y[0,:]+0.01,label='zs')
plt.plot(t, sol.y[2,:],label='zu')

plt.plot(t, road_function(t) ,label='dosso')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
```

<p style="text-align:center;"><img src="/asset/images/scipy/solution_plot.png" alt="solution plot" width="800"></p>


