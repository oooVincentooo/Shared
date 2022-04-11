import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from scipy import signal
import matplotlib.image as mpimg

fig, (ax0,ax, ax2) = plt.subplots(1,3, figsize=(16,9), gridspec_kw={'width_ratios': [1, 2, 1]})

xdata, ydata = [0], [500]

#line, = ax.plot([], [], 'ro')
ax.set_aspect('equal')
ax2.set_aspect('equal')

ax2.set_facecolor("none")
ax0.set_facecolor("none")
ax0.axis('off')
ax2.axis('off')

#Size plot
ax.set_xlim(-600, 600)
ax.set_ylim(-600, 600)
ax0.set_xlim(0, 12)
ax0.set_ylim(12,0)

#plot finish line
ax.plot([0,0],[0,600], color='black', marker='o', linestyle='-',
     linewidth=1, markersize=0     ,zorder=-10           )
ax.plot([-600,600],[0,0], color='lightgray', marker='o', linestyle='-',
     linewidth=1, markersize=0        ,zorder=-10             )


ax.plot([0,0],[-600,0], color='lightgray', marker='o', linestyle='-',
     linewidth=1, markersize=0   ,zorder=-10    )

#Circle with radius 500

circle= plt.Circle((0,0),500, color='red', fill=False)
circle_ani = plt.Circle((0,50000),10, color='red', fill=True)
circle_aq = plt.Circle((0,50000),10, color='red', fill=False)

label_circ_ani = ax0.scatter([10000], [10000], marker='o', s=100, facecolors='red', edgecolors='red', alpha=1.00, label=r"Circle:")


#Circle 500 radius
ax.add_patch(circle)
ax.add_patch(circle_ani)
ax.add_patch(circle_aq)

     

#Square Golden ration S=sqrt(1/phi)
s=1000/np.sqrt((np.sqrt(5)+1)/2)
#print(s)
rect = Rectangle((-np.sqrt(2)*s/2,0),s,s,linewidth=1,edgecolor='blue',facecolor='none', angle=-45)
rect_circ = plt.Circle((0,10000+s/2),10, color='blue', fill=True)
label_circ = ax0.scatter([10000], [10000], marker='o', s=100, facecolors='blue', edgecolors='blue', alpha=1.00, label=r"Curve: $dx/dy$")


rect_aq = plt.Circle((0,10000+s/2),10, color='blue', fill=False, alpha=0.25)
label_aq_ani = ax0.scatter([10000], [10000], marker='o', s=100, facecolors='none', edgecolors='red', alpha=1.00, label=r"Circle:")

label_aq = ax0.scatter([10000], [10000], marker='o', s=100, facecolors='none', edgecolors='blue', alpha=1, label=r"Linear: $\Delta x / \Delta y$")

#Show Image
#image = mpimg.imread('index new.jpg')
#ax2.imshow(image)  



# Add the patch to the Axes
ax.add_patch(rect)
ax.add_patch(rect_circ)
ax.add_patch(rect_aq)

#Animation
frames=31
interval=1000/60
rounds=100

#Pi Aquarian Alalyst
pi_aq=4/1000*s
#print(pi_aq)

timebox=ax.annotate("",(-590,-590), fontsize=16)

print((rounds * (  (frames-1)*(interval/1000  )       )        ))

speed_circ=np.pi*2*500/( (  (frames-1)*(interval/1000  )       )        )
speed_aq=pi_aq*2*500/(   ( (frames-1)*(interval/1000     )    )       )

def init():
    ax.set_aspect('equal')
    ax2.set_aspect('equal')
    
    ax2.set_facecolor("none")
    ax0.set_facecolor("none")
    ax0.axis('off')
    ax2.axis('off') 
    
    #Size plot
    ax.set_xlim(-600, 600)
    ax.set_ylim(-600, 600)
    ax0.set_xlim(0, 12)
    ax0.set_ylim(12,0)
    
    
    #plot finish line
    ax.plot([0,0],[0,600], color='black', marker='o', linestyle='-',
         linewidth=1, markersize=0     ,zorder=-10           )
    ax.plot([-600,600],[0,0], color='lightgray', marker='o', linestyle='-',
         linewidth=1, markersize=0        ,zorder=-10             )  
    ax.plot([0,0],[-600,0], color='lightgray', marker='o', linestyle='-',
         linewidth=1, markersize=0   ,zorder=-10    )    
    
    #Circle with radius 500    
    circle= plt.Circle((0,0),500, color='red', fill=False)    
    rect = Rectangle((-np.sqrt(2)*s/2,0),s,s,linewidth=1,edgecolor='blue',facecolor='none', angle=-45)
    
    legend = ax0.legend(loc='center right', edgecolor="none", fontsize=16,markerfirst=False)
    
    ax.add_patch(rect)    
    ax.add_patch(circle)    
    return rect_aq, rect_circ, circle_ani,timebox, legend,circle_aq

    
def update(frame):
       
    x=500*np.sin(frame*np.pi)
    y=500*np.cos(frame*np.pi)

    x_aq=500*np.sin(frame*pi_aq)
    y_aq=500*np.cos(frame*pi_aq)
    
    circle_ani.set_center((x, y))
    circle_aq.set_center((x_aq, y_aq))
    
    time=1/rounds*frame/2* (frames-1)*(interval/1000  )  
    rnds=frame/2
    
    distance_circ=speed_circ*frame/2* (frames-1)*(interval/1000  )  
    yr_circ= -np.sqrt(2)*s/2*signal.sawtooth(1/(4*s)*distance_circ*2*np.pi, 0.5)
    xr_circ= -np.sqrt(2)*s/2*signal.sawtooth(1/(4*s)*(distance_circ-s)*2*np.pi, 0.5)
    rect_circ.set_center((xr_circ, yr_circ))
    label_circ.set_label("Square $\pi$: \nDistance: "+ str("{:09.2f}".format((distance_circ))) +  "\nSpeed: " + str("{:.4f}".format(round(distance_circ/time,4)) )  +"\n"    )    
    
    label_circ_ani.set_label("Circle $\pi$: \nDistance: "+ str("{:09.2f}".format((distance_circ)))+  "\nSpeed: " + str("{:.4f}".format(round(distance_circ/time,4)) +"\n"  ) )     
 
    
    distance_aq=speed_aq*frame/2* (frames-1)*(interval/1000  )  
    yr_aq= -np.sqrt(2)*s/2*signal.sawtooth(1/(4*s)*distance_aq*2*np.pi, 0.5)
    xr_aq= -np.sqrt(2)*s/2*signal.sawtooth(1/(4*s)*(distance_aq-s)*2*np.pi, 0.5)
    rect_aq.set_center((xr_aq, yr_aq))
    label_aq_ani.set_label("Circle $4/ \sqrt{\phi}$: \nDistance: "+ str("{:09.2f}".format((distance_aq)))+  "\nSpeed: " + str("{:.4f}".format(round(distance_aq/time,4)) +"\n"  ) )     
 
    label_aq.set_label("Square $4/ \sqrt{\phi}$: \nDistance: "+ str("{:09.2f}".format((distance_aq)))+  "\nSpeed: " + str("{:.4f}".format(round(distance_aq/time,4)) +"\n"  ) )     

    timebox.set_text("\nTime: " + str("{:.4f}".format(time,4))   + " seconds\nRounds: " +  str("{:.2f}".format(rnds)) + " of " + str(int(rounds))  +" rounds" )

    legend = ax0.legend(loc='center right', edgecolor="none", fontsize=16,markerfirst=False)
    
    return rect_aq, rect_circ, circle_ani,timebox, legend,circle_aq

ID=np.linspace(0.01,rounds*2, frames)
final=np.full(( int(15/(interval/1000) )         ),ID[-1])
ID=np.append(ID, final)

plt.tight_layout()
anim = animation.FuncAnimation(fig, update, frames=ID, interval=interval, init_func=init, blit=True, repeat=False )


#Save video
#f = r"c://temp/500 s 60 fps 100 rounds Aquarian OOOVincentOOO.avi" 
#writervideo = animation.FFMpegWriter(fps=1000/interval)
#anim.save(f, writer=writervideo, dpi = 300)

plt.show()


print("finished")
