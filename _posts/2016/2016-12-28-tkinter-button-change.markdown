---
layout: post
title:  "Tkinter button changes when pressed"
date:   2016-12-28 12:30:00 +0100
categories: python-GUI
---

Tkinter button that changes appearance when pressed/released. The files *up.gif* and *down.gif* can be downloaded from below.


{% highlight python %}
try:
    # Python2
    import Tkinter as tk
except ImportError:
    # Python3
    import tkinter as tk

app = tk.Tk()

image_down = tk.PhotoImage(file='down.gif')
def pressed(event):
    button.config(image=image_down)

image_up = tk.PhotoImage(file='up.gif')
def released(event):
    button.config(image=image_up)


button = tk.Button(app, image=image_up, bd=0)
button.grid(row=1, column=1)
button.bind('<Button-1>', pressed)
button.bind('<ButtonRelease-1>', released)

app.geometry("135x60")
app.mainloop()
{% endhighlight %}


{% include _images/{{ page.date | date: '%Y-%m-%d'}}/tkinter_example.html  %}

