---
layout: post
title:  "Tkinter with matplotlib"
date:   2016-09-05 12:30:00 +0100
categories: python-GUI
---

{% highlight python %}
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from tkinter import ttk
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
matplotlib.use("TkAgg")


class ConoReader(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "Random noise")

        # OUTPUT FIGURE
        self.f = Figure(figsize=(5,3), dpi=100)
        self.a = self.f.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.f, self)

        self.canvas.get_tk_widget().pack()
        self.a.imshow(np.random.random([100,100]), cmap="magma")
        self.canvas.show()

        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        self.toolbar.update()
        self.canvas._tkcanvas.pack()

        # BUTTON
        button = ttk.Button(self, text=">>", command=self.nextImage)
        button.pack()

    def nextImage(self):
        self.a.imshow(np.random.random([100,100]), cmap="viridis")
        self.canvas.show()

app = ConoReader()
app.geometry("600x400")
app.mainloop()
{% endhighlight %}


{% include _images/{{ page.date | date: '%Y-%m-%d'}}/tkinter_matplotlib.html  %}

