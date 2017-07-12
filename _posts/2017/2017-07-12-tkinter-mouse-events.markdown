---
layout: post
title:  "Tkinter mouse events"
date:   2017-07-12 08:30:00 +0100
categories: python-GUI
---

{% highlight python %}
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class App:
    def __init__(self, master):
        frame = tkinter.Frame(master)
        
        fig = Figure()
        ax = fig.add_subplot(111)
        self.line, = ax.plot([x**2 for x in range(10)])
        self.canvas = FigureCanvasTkAgg(fig, master=master)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        self.canvas.callbacks.connect('button_press_event', self.callback)
        frame.pack()

    def callback(self, event):
        if event.dblclick:
            print(event, event.x, event.y)

root = tkinter.Tk()
app = App(root)
root.mainloop()
{% endhighlight %}