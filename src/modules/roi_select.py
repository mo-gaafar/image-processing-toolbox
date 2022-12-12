""" This is a module with functions to select a region of interest (ROI) from an image.
Created on 2022/11
Author: M. Nasser Gaafar

Functions:
    show_roi_window(self): shows the ROI selection window
    select_roi(self): selects the ROI and updates the image

"""

from matplotlib.widgets import RectangleSelector
import numpy as np
import matplotlib.pyplot as plt


def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
    print(" The button you used were: %s %s" %
          (eclick.button, erelease.button))


def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)


fig, current_ax = plt.subplots()                 # make a new plotting range
N = 100000                                       # If N is large one can see
x = np.linspace(0.0, 10.0, N)                    # improvement by use blitting!

plt.plot(x, +np.sin(.2*np.pi*x), lw=3.5, c='b', alpha=.7)  # plot something
plt.plot(x, +np.cos(.2*np.pi*x), lw=3.5, c='r', alpha=.5)
plt.plot(x, -np.sin(.2*np.pi*x), lw=3.5, c='g', alpha=.3)

print("\n      click  -->  release")

# drawtype is 'box' or 'line' or 'none'
toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                       drawtype='box', useblit=True,
                                       # don't use middle button
                                       button=[1, 3],
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)
plt.connect('key_press_event', toggle_selector)
plt.show()


def show_roi_window(self):
    """ Shows the ROI selection matplot window
    """

    # create a new figure
    self.roi_fig = plt.figure()

    # create a new axis
    self.roi_ax = self.roi_fig.add_subplot(111)

    # plot the image
    self.roi_ax.imshow(self.image1.get_image())

    # create a new rectangle selector
    self.roi_selector = RectangleSelector(
        self.roi_ax, self.select_roi, drawtype='box', useblit=True, button=[1, 3], minspanx=5, minspany=5, spancoords='pixels', interactive=True)

    plt.connect('key_press_event', self.toggle_selector)
    # show the figure
    plt.show()

def display_roi_preview(self):
    pass