import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.patches as mpatches


fig= plt.figure(figsize=(16, 8), constrained_layout=True)

widths = [4,4,8]
heights = [4,4]
gs=fig.add_gridspec(2,3,width_ratios=widths, height_ratios=heights)

ax1a=fig.add_subplot(gs[0,0])
ax2a=fig.add_subplot(gs[0,1])
ax3a=fig.add_subplot(gs[0,2])
ax4a=fig.add_subplot(gs[1,0])
ax5a=fig.add_subplot(gs[1,1])
ax6a=fig.add_subplot(gs[1,2])

   
def G1(C,ax1a):
    
    R=1/(2*np.sin(np.pi/C))
    Rs=1/(2*np.tan(np.pi/C))
    r=R+Rs-np.round(C/(np.pi),0)
    alpha=np.abs(2*np.arctan(0.5/r))
    y=np.cos(alpha/2)
    x=np.sin(alpha/2)

    #circle1=plt.Circle((R, 0), R, color='black', linestyle='--',alpha=1,fill=False)
    #ax1a.add_patch(circle1)


    e=r

    circle1=plt.Circle((R, 0), R, color='black', linestyle='--',alpha=1,fill=False)
    ax1a.add_patch(circle1)

    #Coins Plot
    props={'facecolor':'white', 'alpha':0, 'pad':10, 'ec':'none'}
    text= r'$\alpha$' + '=' +            str("{:.4f}".format(np.degrees(alpha)))    + r'$^{\circ}$'  + '\n' + r'$\varepsilon$' + '=' +    str("{:.4f}".format(r))
    #ax1a.text(0.05, 0.95, text  , transform=ax1a.transAxes, fontsize=12,verticalalignment='top', bbox=props)

    ax1a.annotate(text, # this is the text
                     (-0.8,0.8), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,0), # distance from text to points (x,y)
                     ha='left',
                     va='top',
                     fontsize=12,
                     color='black',
                     rotation=0) # horizontal alignment can be left, right or center

    #text='Discrete:\n' + r'$C=$' + str(np.round(C,4))  + '\n' +  r'$R=$' + str(np.round(C/np.pi,0))   + '\n'  +  r'C/R=' +  str(np.round(C/np.round(C/np.pi,0),4) )

    text= r'$C=$' + str("{:.0f}".format(C))  + '\n' +  r'$D=$' + str("{:.0f}".format(C/np.pi))   + ' ('   +           str("{:.4f}".format(((C/np.pi))  ))  + ')'  + '\n'  +  r'C/D=' +  str(np.round(C/np.round(C/np.pi,0),4) )

    ax1a.annotate(text, # this is the text
                     (-0.8,-0.5), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,0), # distance from text to points (x,y)
                     ha='left',
                     va='top',
                     fontsize=12,
                     color='black',
                     rotation=0) # horizontal alignment can be left, right or center

    circle2=plt.Circle((e+R-Rs, 0), 0.5, color='r', alpha=0.2)
    ax1a.add_patch(circle2)
    ax1a.plot([0+R-Rs,e+R-Rs,0+R-Rs,0+R-Rs],[0.5,0,-0.5,-0.5], marker='.', color='black', linestyle='-', linewidth=1, markersize=5,zorder=10,alpha=1)
    ax1a.plot([e+R-Rs],[0], marker='.', color='r', linestyle='-', linewidth=0, markersize=5,zorder=10,alpha=1)


    circle3=plt.Circle((e+1+R-Rs, 0), 0.5, color='r', alpha=0.2)
    ax1a.add_patch(circle3)
    ax1a.plot([e+R-Rs],[0], marker='.', color='black', linestyle='-', linewidth=0, markersize=5,zorder=10,alpha=1)

    if e<0:
        pac = mpatches.Arc([e+R-Rs,0], 0.2, 0.2, angle=0, theta1=0-np.degrees(alpha/2), theta2=0+np.degrees(alpha/2),linewidth=0.3)
        ax1a.add_patch(pac)

    else:
        pac = mpatches.Arc([e+R-Rs,0], 0.2, 0.2, angle=0, theta1=180-np.degrees(alpha/2), theta2=180+np.degrees(alpha/2),linewidth=0.3)
        ax1a.add_patch(pac)



    if C%2==0:


        circle4=plt.Circle((0, 0), 0.5, color='black', alpha=0.2)
        ax1a.add_patch(circle4)

        x=R-R*np.cos(2*np.pi/C)
        y=R*np.sin(2*np.pi/C)   

        circle5=plt.Circle((x, y), 0.5, color='black', alpha=0.2)
        ax1a.add_patch(circle5)   

        circle6=plt.Circle((x, -y), 0.5, color='black', alpha=0.2)
        ax1a.add_patch(circle6)   

        ax1a.plot([x],[y], marker='.', color='black', linestyle='-', linewidth=0, markersize=5,zorder=10,alpha=1)
        ax1a.plot([x],[-y], marker='.', color='black', linestyle='-', linewidth=0, markersize=5,zorder=10,alpha=1)
        ax1a.plot([0],[0], marker='.', color='black', linestyle='-', linewidth=0, markersize=5,zorder=10,alpha=1)


    else:

        x=R-R*np.cos(np.pi/C)
        y=R*np.sin(np.pi/C)   

        circle5=plt.Circle((x, y), 0.5, color='black', alpha=0.2)
        ax1a.add_patch(circle5)   

        circle6=plt.Circle((x, -y), 0.5, color='black', alpha=0.2)
        ax1a.add_patch(circle6)       

        ax1a.plot([x],[y], marker='.', color='black', linestyle='-', linewidth=0, markersize=5,zorder=10,alpha=1)
        ax1a.plot([x],[-y], marker='.', color='black', linestyle='-', linewidth=0, markersize=5,zorder=10,alpha=1)



    ax1a.plot([e+R-Rs,e+R-Rs],[-1,1],linestyle='--', color='black', linewidth=0.7, markersize=0,zorder=10,alpha=0.5 )
    ax1a.plot([0,0],[-1,1],linestyle='-', color='black', linewidth=0.7, markersize=0,zorder=10,alpha=1 )
    #ax2a.plot([R-Rs,R-Rs],[-1,1],linestyle='--', color='blue', linewidth=0.5, markersize=0,zorder=10,alpha=0.5 )
    ax1a.set_title('Coin circle',fontsize=14)

    #ax1a.axis('equal')
    ax1a.axes.set_ylim([-.9,.9])
    ax1a.set_yticks([-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8]) 
    ax1a.axes.set_xlim([-.9,.9])
    ax1a.set_xticks([-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8]) 




def G2(C,ax2a):
    
    R=1/(2*np.sin(np.pi/C))
    Rs=1/(2*np.tan(np.pi/C))
    r=R+Rs-np.round(C/(np.pi),0)
    alpha=np.abs(2*np.arctan(0.5/r))
    y=np.cos(alpha/2)
    x=np.sin(alpha/2)
    #Alpha Circles

    #Circle 90 degrees
    circle1=plt.Circle((0.5, 0), 1/np.sqrt(2), color='red', linestyle='-',alpha=0.5,fill=False,linewidth=0.8)
    ax2a.add_patch(circle1)
    ax2a.plot([0,0.5,0],[0.5,0,-0.5],marker='.',markersize=0,color='black',linewidth=0.1,zorder=1)
    ax2a.plot([0.5],[0],marker='.',markersize=5,color='black',linewidth=0.1)

    pac = mpatches.Arc([0.5,0], 0.15, 0.15, angle=90, theta1=45, theta2=45+90,color='black',linewidth=0.3)
    ax2a.add_patch(pac)

    ax2a.annotate(r'$\alpha$=90$^{\circ}$', # this is the text
                     (0.5,0.1), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(-2,2), # distance from text to points (x,y)
                     ha='right',
                     fontsize=10,
                     color='black',
                     rotation=90) # horizontal alignment can be left, right or center


    #ax2a.annotate(r'$\varepsilon$=0.5', # this is the text
    #                 (0,-0.85), # this is the point to label
    #                 textcoords="offset points", # how to position the text
    #                 xytext=(0,5), # distance from text to points (x,y)
    #                 ha='left',
    #                 fontsize=10,
    #                 color='black',
    #                 rotation=0) # horizontal alignment can be left, right or center

    #------------------------------



    
    #Circle alpha degrees
    #alpha=np.log(2)+np.pi/2
    absalpha=np.abs(alpha)
    epsilon=1/(2*np.tan(alpha/2))
    R=1/(2*np.sin(alpha/2))

    circle1=plt.Circle((epsilon, 0), R, color='black', linestyle='-',alpha=0.15,fill=True,zorder=1,linewidth=0.3)
    ax2a.add_patch(circle1)

    ax2a.plot([0,epsilon,0],[0.5,0,-0.5],marker='.',markersize=0,color='black',linewidth=1,zorder=1)
    ax2a.plot([epsilon],[0],marker='.',markersize=5,color='black',linewidth=0.1)

    pac = mpatches.Arc([epsilon,0], 0.15, 0.15, angle=90, theta1=90-np.degrees(alpha/2), theta2=   180 -90+np.degrees(alpha/2)   ,color='black',linewidth=0.3)
    ax2a.add_patch(pac)

    ax2a.annotate(r'$\alpha$=' +     str("{:.4f}".format(np.degrees(alpha)))        +         r'$^{\circ}$', # this is the text
                     (-0.5,0.8), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,0), # distance from text to points (x,y)
                     ha='left',
                     va='top',
                     fontsize=12,
                     color='black',
                     rotation=0) # horizontal alignment can be left, right or center


    ax2a.plot([0,epsilon],[-0.8,-0.8],marker='.',markersize=5,color='black',linewidth=0.5,zorder=1)

    ax2a.annotate(r'$\varepsilon$=' +    str("{:.4f}".format(epsilon))  , # this is the text
                     (0,-0.8), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,5), # distance from text to points (x,y)
                     ha='left',
                     fontsize=10,
                     color='black',
                     rotation=0) # horizontal alignment can be left, right or center

    #------------------------------



    #Circle 180 degrees
    circle1=plt.Circle((0, 0), 0.5, color='green', linestyle='-',alpha=0.5,fill=False,linewidth=1,zorder=-10)
    ax2a.add_patch(circle1)

    #circle1=plt.Circle((0, 0), 0.5, color='black', linestyle='-',alpha=1,fill=False,linewidth=0.3,zorder=-10)
    #ax2a.add_patch(circle1)

    ax2a.plot([0,0,0],[-0.5,0,0.5],marker='.',markersize=5,color='black',linewidth=0.3,zorder=-100)
    pac = mpatches.Arc([0,0], 0.15, 0.15, angle=180, theta1=-90, theta2=90,linewidth=0.2)
    ax2a.add_patch(pac)

    ax2a.annotate(r'$\alpha$=180$^{\circ}$', # this is the text
                     (0,0.1), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(-2,0), # distance from text to points (x,y)
                     ha='right',
                     fontsize=10,
                     color='black',
                     rotation=90) # horizontal alignment can be left, right or center

    ax2a.plot([0,0.5],[10,10],marker='.',markersize=5,color='black',linewidth=0.5,zorder=0.8)
    ax2a.set_title(r'Error angle $\alpha$',fontsize=14)

    #------------------------------

    #alpha=np.log(2)+np.pi/2
    #epsilon=1/(2*np.tan(alpha/2))
    #R=1/(2*np.sin(alpha/2))
    ax2a.axhline(y=0, color='k',linewidth=0.5)
    ax2a.axvline(x=0, color='k',linewidth=0.5)
    ax2a.axvline(x=0.5, color='k',linewidth=0.5)
    ax2a.axvline(x=epsilon, color='k',linewidth=0.5)

    ax2a.axis('equal')
    ax2a.axes.set_ylim([-0.9,0.9])
    ax2a.set_yticks([-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8])
    ax2a.axes.set_xlim([-0.55,1.25])
    ax2a.set_xticks([-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,1.2]) 
    #ax0a.set_yticks(np.arange(-1, 1.2, step=0.2)) 




    #Alpha plot
def G3(p,ax3a):
    
    #C=np.arange(0,p+1,1)+2
    #Ro=1/(2*np.sin(np.pi/C))
    #Ri=1/(2*np.tan(np.pi/C))
    #D=np.round(C/np.pi,0)
    #epsilon=Ro+Ri-D
    
    range=100

    C=np.arange(0,p+range/2+1,1)+2
    
    Cc=np.arange(0,p+range/2+1,0.01)+2
        
    D=np.round(C/np.pi,0)
    #D=C/np.pi

    R=1/(2*np.sin(np.pi/C))
    Rs=1/(2*np.tan(np.pi/C))
    epsilon=R+Rs-D


    alpha=np.degrees(2*np.arctan(0.5/epsilon))

    alpha=2*np.arctan( 1/ ( 2*  ( epsilon    )          )                      )
    alpha=np.degrees(alpha)
    ax3a.plot(C,(alpha),linestyle='-',marker='.',markersize=3,color='black',label=r'$\alpha(c)$' + ' discrete' ,linewidth=0,alpha=1  )

    alphac=2*np.arctan( 1/ ( 2*  ( Cc/np.pi - np.round(Cc/np.pi,0)      )          )                      )
    alphac=np.degrees(alphac)
    ax3a.plot(Cc,(alphac),linestyle='-',marker='.',markersize=0,color='black',label=r'$\alpha(c)$' + ' continuous' ,linewidth=0.2,alpha=1  )

    ax3a.axes.set_ylim([-260,260])
    ax3a.axes.set_xlim([p-range/2,p+range/2])

    #mean=np.log(2)+np.pi/2
    #ax3a.plot([p-200+2,p+2],[mean,mean],linestyle='-',marker='.',markersize=0,color='red' ,linewidth=1,alpha=1  )
    #ax3a.plot([p-200+2,p+2],[-mean,-mean],linestyle='-',marker='.',markersize=0,color='red' ,linewidth=1,alpha=1  )

    ax3a.plot([p-75+2,p+2+75],[np.mean(np.abs(alpha)),np.mean(np.abs(alpha))],linestyle='-',marker='.',markersize=0,color='red' ,linewidth=0.8,alpha=1  )
    ax3a.plot([p-75+2,p+2+75],[-np.mean(np.abs(alpha)),-np.mean(np.abs(alpha))],linestyle='-',marker='.',markersize=0,color='red' ,linewidth=0.8,alpha=1  )


    #print(np.degrees(np.mean(np.abs(alpha))))

    ax3a.annotate( r'$\overline{\alpha}=log(2)+\pi/2$' + '=2.2639...=129.7144...$^{\circ}$ (theory)\n'  +     r'$\overline{\alpha}$=' +
                   str('{:.4f}'.format(  (np.mean(np.abs(alpha)))                   )  ) + r'$^{\circ}$'  +  ' (measured)',     # this is the text
                     (p-range/2,255), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(5,0), # distance from text to points (x,y)
                     ha='left',
                     va='top',
                     fontsize=12,
                     color='black') # horizontal alignment can be left, right or center


    ax3a.fill([(p+2),(p+2+np.pi),(p+2+np.pi),(p+2)],[-260,-260,260,260],color='black', alpha=0.05  ,zorder=-10  )

    ax3a.annotate(r'$\Delta c=\pi$', # this is the text
                 (p+2,-250), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(1,0), # distance from text to points (x,y)
                 ha='left',
                 fontsize=10,
                 color='black') # horizontal alignment can be left, right or center


    ax3a.fill([p-range/2,p+range/2,p+range/2,p-range/2],[90,90,180,180],color='blue', alpha=0.05  ,zorder=-10 )
    ax3a.fill([p-range/2,p+range/2,p+range/2,p-range/2],[-90,-90,-180,-180],color='blue', alpha=0.05  ,zorder=-10  )

    ax3a.axvline(x=p+2, color='k',linewidth=0.5)    
    ax3a.axhline(y=0, color='k',linewidth=0.5)
    
    ax3a.legend(loc ="lower right",fontsize=10)
    ax3a.set_xlabel('circumference: c',fontsize=12)
    ax3a.set_ylabel(r'$\alpha$',fontsize=12)
    #ax3a.set_xticks(np.linspace(p-0.75*range/2,p+0.75*range/2,6)) 
    bin=np.round(range/10,0)
    ax3a.set_xticks([2+p-4*bin,2+p-3*bin,2+p-2*bin,2+p-bin,2+p,2+p+bin,2+p+2*bin,2+p+3*bin,2+p+4*bin])
    #ax3a.get_xaxis().set_visible(False)
    ax3a.set_title('Error angle: ' + r'$\alpha$',fontsize=14)    

    #----------------------------------------------


def G45(p,ax4a,ax5a):


    #Histogram


    C=np.arange(0,p+1,1)+2
    D=np.round(C/np.pi,0)
    #D=C/np.pi

    R=1/(2*np.sin(np.pi/C))
    Rs=1/(2*np.tan(np.pi/C))
    r=R+Rs-D


    alpha=np.degrees(2*np.arctan(0.5/r))

    #x = np.where(x<0, 0., x*10)
    #alpha=np.where(alpha<0,-alpha,alpha)
    alpha=np.abs(alpha)

    #r1=2*R-D
    r1=r
    #r1=np.where(r1<0,-r1,r1)
    r1=np.abs(r1)
    #alpha1=np.degrees(2*np.arctan(0.5/r1))


    bins=30

    ax4a.set_title('PDF error distance ' + r'$\epsilon$'    +  '\nHistogram ' + str(p+2)  +   ' coin circles',fontsize=14)
    b=np.linspace(0,0.5,bins)
    ax4a.hist(r1,bins=b,rwidth=1,density=True,alpha=0.75,label='statistical')
    ax4a.plot([0,0.5],[2,2],linestyle='-',color='red',label='theoretical')

    #ax1a.axes.set_xlim([-1.2,1.2])
    ax4a.axes.set_ylim([0,2.5])
    ax4a.axes.set_xlim([-0.05,0.55])


    ax4a.legend(loc ="lower right",fontsize=8)
    ax4a.grid(b=True, which='major', color='#666666', linestyle='-', zorder=0)
    ax4a.set_xlabel('error distance: ' + r'$\epsilon$',fontsize=12)
    ax4a.set_ylabel('Density',fontsize=12)

   
    b=np.linspace(90,180,bins)
    ax5a.hist(alpha,bins=b,rwidth=1,density=True,alpha=0.75,label='statistical')

    meant=np.log(2)+np.pi/2
    mean=np.mean(np.radians(alpha))
    stdevt=np.sqrt(-4*0.915965594177219-(np.log(4))**2/4+np.pi*np.log(4))
    stdev=np.std(np.radians(alpha))
    
    #mean=2*np.degrees(0.5*np.log(4*0.5**2+1)+2*0.5*np.arctan(1/(2*0.5)))
    epsilon=np.linspace(0,0.5,500)

    alpha=np.linspace(90,180,100)


    #mean=0.5*np.log(4*0.5**2+1)+2*0.5*np.arctan(1/(2*0.5))

    pdf=np.pi/90*(1/(np.tan(np.radians(alpha)/2)**2)+1)/4

    ax5a.plot((alpha),pdf,linestyle='-',color='red',label='theoretical')
    #ax2a.plot([90,180],[1/90,1/90],linestyle='-',color='blue')
    #ax2a.text(165,0.92* 1/90, '1/90=0.0111...')
    ax5a.plot([np.degrees(mean),np.degrees(mean)],[0,0.02],linestyle='-',color='green',linewidth=1)
    ax5a.set_xlim(ax5a.get_xlim()[::-1])


    #print(np.degrees(stdev))
    ax5a.plot([np.degrees(mean-stdev),np.degrees(mean-stdev)],[0,0.02],linestyle='--',color='green',linewidth=1)
    ax5a.plot([np.degrees(mean+stdev),np.degrees(mean+stdev)],[0,0.02],linestyle='--',color='green',linewidth=1)

    #ax5a.text(0.99*np.degrees(meant),0.88*0.02,'mean ' + r'$\overline{\alpha}$'    +   ':\n' + str(np.round(np.degrees(mean),3))  + r'$^{\circ}$')
    #ax5a.text(0.99*np.degrees(meant+stdevt),0.88*0.02,'std:\n' +  r'$\overline{\alpha}$' + '+'    + str(np.round(np.degrees(stdev),3)) + r'$^{\circ}$')
    #ax5a.text(0.99*np.degrees(meant-stdevt),0.88*0.02,'std:\n' +  r'$\overline{\alpha}$' + '-'    + str(np.round(np.degrees(stdev),3)) + r'$^{\circ}$')

    ax5a.annotate('mean ' + r'$\overline{\alpha}$'    +   ':\n' + str(np.round(np.degrees(mean),3))  + r'$^{\circ}$', # this is the text
             (0.99*np.degrees(meant),0.88*0.02), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(-5,0), # distance from text to points (x,y)
             ha='right',
             fontsize=10,
             color='black') # horizontal alignment can be left, right or center
 
    ax5a.annotate('std:\n' +  r'$\overline{\alpha}$' + '+'    + str(np.round(np.degrees(stdev),3)) + r'$^{\circ}$', # this is the text
             (0.99*np.degrees(meant+stdevt),0.88*0.02), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(-5,0), # distance from text to points (x,y)
             ha='right',
             fontsize=10,
             color='black') # horizontal alignment can be left, right or center

    ax5a.annotate('std:\n' +  r'$\overline{\alpha}$' + '-'    + str(np.round(np.degrees(stdev),3)) + r'$^{\circ}$', # this is the text
         (0.99*np.degrees(meant-stdevt),0.88*0.02), # this is the point to label
         textcoords="offset points", # how to position the text
         xytext=(-5,0), # distance from text to points (x,y)
         ha='right',
         fontsize=10,
         color='black') # horizontal alignment can be left, right or center
    
    
    
    ax5a.set_xlabel('angle: ' + r'$\alpha$'+  r'$^{\circ}$',fontsize=12)
    ax5a.set_ylabel('Density',fontsize=12)
    ax5a.axes.set_ylim([0,.02])
    ax5a.xaxis.set_ticks(np.arange(90,180.1,15))

    ax5a.set_title('PDF TwoTwo Angle \nHistogram ' + str(p+2)  +   ' coin circles',fontsize=14)
    ax5a.grid(b=True, which='major', color='#666666', linestyle='-', zorder=0)
    ax5a.legend(loc ="lower right",fontsize=8)


    #---------------------------------------------------------------------

def G6(p,N,ax6a):

    #Pi estimate


    C=np.arange(0,p+1,1)+2
    Ro=1/(2*np.sin(np.pi/C))
    Ri=1/(2*np.tan(np.pi/C))
    #D=np.round(2*Ro,0)
    D=np.round(C/np.pi,0)
    Coins=C+D
    epsilon=Ro+Ri-D

    alpha=(2*np.arctan(1/(2*epsilon)))

    #dD=np.abs(2*Ro-np.round(2*Ro,0))
    #f=np.where(dD>=0.48)

    #dDf=dD[f]

    var=-4*0.915965594177-(np.log(4))**2/4+np.pi*np.log(4)
    stdev=np.sqrt(var)


    pia=2*(np.abs(alpha)-np.log(2))
    piac=np.cumsum(pia)/(C-1)

    meanalpha=np.log(2)+np.pi/2
    C1=np.arange(0,N,1)+2
    UB=meanalpha+2*stdev/(np.sqrt((C1-1)))
    LB=meanalpha-2*stdev/(np.sqrt((C1-1)))

    UBpi=2*(UB-np.log(2))
    LBpi=2*(LB-np.log(2))

    x=C1
    xf=np.flip(x)
    yf=np.flip(UBpi)

    ax6a.fill(np.append(x,xf),np.append(LBpi,yf),color='#1f77b4',alpha=0.05,label=r'$95 \% $,  2std/$\sqrt{n}$',zorder=-50)

    ax6a.plot(C,pia,linestyle='-',color='red', linewidth=0,marker='.',markersize=2,zorder=-10,alpha=0.2 )
    ax6a.plot(C,piac,linestyle='-',color='red',alpha=1, linewidth=0.8,label=r'$\pi$ mean: $\overline{\alpha}=$log(2)+$\pi/2$'  ,zorder=10 )




    D=np.round(C/np.pi,0)
    pidis=C/D
    pic=np.cumsum(pidis)/(C-1)

    ax6a.plot(C,pidis,linestyle='-',color='black',marker='.',markersize=2, alpha=0.2,linewidth=0   ,zorder=10 )
    ax6a.plot(C,pic,linestyle='-',color='black',alpha=1, linewidth=0.8,label=r'$\pi$ discrete: $C/D$'   ,zorder=50 )
    
    
    x=np.arange(0,p+1,1)+2
    ye=0.5*np.random.random_sample((p+1,))
    ya=2*np.arctan(1/(2*ye))
    ypi=2*(ya-np.log(2))
    ypic=np.cumsum(ypi)/(x-1)
    
    
    ax6a.plot(x,ypic,linestyle='-',color='#1f77b4',alpha=0.5, linewidth=0.8,label=r'$\pi$ sampled: random (theory)'   ,zorder=-10 )
    

    if p!=0:
        ax6a.annotate(r'$\pi$=' +  str("{:.4f}".format( piac[-1] ))   +  r' (total mean $\alpha$)' + '\n'  +    r'$\pi$=' +  str("{:.4f}".format( pic[-1] ))   +  ' (total mean C/D)'  + '\n'   +    r'$\pi$=' +  str("{:.4f}".format( ypic[-1] ))   +  ' (random sample)'                                  +  '\n' + r'$C=$' + str("{:.0f}".format(p+2))        , # this is the text
                         (2,1.15*np.pi), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(4,-5), # distance from text to points (x,y)
                         ha='left',
                         va='top',
                         fontsize=12,
                         color='black') # horizontal alignment can be left, right or center


    var=-4*0.915965594177+ 1/4*np.pi*(np.pi+ np.log(16))
    stdev=np.sqrt(var)
    UB=meanalpha+stdev/(np.sqrt(C-1))
    LB=meanalpha-stdev/(np.sqrt(C-1))

    #ax6a.plot([0,N],[np.pi,np.pi],linestyle='-',linewidth=1,color='blue')
    ax6a.axhline(y=np.pi, color='k',linewidth=1)

    ax6a.set_xlabel('circumference: c',fontsize=12)
    ax6a.set_ylabel(r'$\pi$',fontsize=12)
    ax6a.axes.set_ylim([0.85*np.pi,1.15*np.pi])
    ax6a.axes.set_xlim([2,N])

    ax6a.yaxis.set_ticks(np.arange(  0.9*np.pi , 1.1*np.pi,np.pi/20))

    #ax1a.xaxis.set_ticks(np.arange(90,180.1,15))
    ax6a.set_title(r'$\pi$' + ' from mean error angle ' + r'$\alpha$ and discrete',fontsize=14)
    ax6a.grid(b=True, which='major', color='#666666', linestyle='-', zorder=0)
    ax6a.legend(loc ="lower right",fontsize=10)
    #ax6a.set_xticks(np.arange25) 
    ax6a.set_xscale('log')

    #--------------------------------------------------------------------

N=60000-2
#for p in range(N):

p=60000-2
C=p+2

ax1a.clear()
ax2a.clear()
ax3a.clear()
ax4a.clear()
ax5a.clear()
ax6a.clear()

G1(C,ax1a)
G2(C,ax2a)
G3(p,ax3a)
G45(p,ax4a,ax5a)
G6(p,N,ax6a)

plt.savefig('TwoTwo_Cockpit' + str(f"{p:04d}") + '.jpg', bbox_inches='tight',dpi=150)