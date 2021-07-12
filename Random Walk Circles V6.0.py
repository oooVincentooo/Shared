import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import fractions

#Open pyplot in separate interactive window
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

fig= plt.figure(figsize=(30,9)) 

widths = [3,3,3,3,3,3,3,3,3,3]
heights = [3,3,3]
gs=fig.add_gridspec(3,10,width_ratios=widths, height_ratios=heights)
 
for plot in range(5):
    fig.add_subplot(gs[0,plot*2+0:plot*2+1])
    fig.add_subplot(gs[1,plot*2+0:plot*2+1])
    fig.add_subplot(gs[1,plot*2+1:plot*2+2])
    fig.add_subplot(gs[0,plot*2+1:plot*2+2])

ax_list=fig.axes

def plotclear(ax_list):

    for axs in range(5):
        plot= axs*4
        ax_list[plot+0].clear()
        ax_list[plot+1].clear()
        ax_list[plot+2].clear()
        ax_list[plot+3].clear()       

def plotsetup(ax_list):

    for axs in range(5):
        plot= axs*4
        ax_list[plot+0].axes.set_xlim([-0.2,0.2])
        ax_list[plot+0].set_ylim([-0.05,0.35])

        #ax_list[plot+0].axis('equal')
        ax_list[plot+1].axis('equal')
        ax_list[plot+2].axis('equal')       

        ax_list[plot+1].axis('off')
        ax_list[plot+2].axis('off')
        ax_list[plot+3].axis('off')

def plots(plot,p,n,steps,parray):
    N=np.arange(0,steps)
    t=N/n

    px=parray[p-2][0]
    py=parray[p-2][1]

    x=1/(steps)*np.cos((px+t*2)*np.pi)
    x=np.append(x,x[0])

    y=1/(steps)*np.sin((py+t*2)*np.pi)
    y=np.append(y,y[0])

    xc=np.cumsum(x)
    yc=np.cumsum(y)

    xd=np.diff(xc[:-1])
    yd=np.diff(yc[:-1])
    dr=np.sqrt(xd**2+yd**2)
    circum=np.sum(dr)

    minx=np.min(xc[:n]); miny=np.min(yc[:n])
    maxx=np.max(xc[:n]); maxy=np.max(yc[:n])

    start=int(steps*1/8)

    ax_list[4*plot+0].plot(xc[:n],yc[:n],linewidth=0.15,color='black')

    arrowdic=dict(arrowstyle="-",lw=0.5,color="black",shrinkA=10, shrinkB=5)

    ax_list[4*plot+0].annotate('A', xy=(xc[n],yc[n]),  xycoords='data', fontsize=16,
            xytext=(0.0,-0.3), textcoords='axes fraction', color='black', arrowprops=arrowdic)   

    ax_list[4*plot+1].plot(xc[:steps],yc[:steps],linewidth=0.15,color='black')
    
    start=int(1/8*steps-200)
    stop=int(1/8*steps)
        
    minsqrx=np.min(xc[start:stop]); minsqry=np.min(yc[start:stop])
    maxsqrx=np.max(xc[start:stop]); maxsqry=np.max(yc[start:stop])        
        
    rect = patches.Rectangle((minsqrx,minsqry), maxsqrx-minsqrx,maxsqry-minsqry, linewidth=0.5, edgecolor='red', facecolor='none')
    
    # Add the patch to the Axes
    ax_list[4*plot+1].add_patch(rect)
    
    arrowdic=dict(arrowstyle="-",lw=0.5,color="red",shrinkA=10, shrinkB=5)
    
    ax_list[4*plot+1].annotate('B', xy=(maxsqrx,minsqry),  xycoords='data', fontsize=16,
            xytext=(1.2,0.05), textcoords='axes fraction', color='red', arrowprops=arrowdic)
    
    text=('200 steps, 1/8 total')
    ax_list[4*plot+2].text(0,0,text, fontsize=12, horizontalalignment='left',
                           verticalalignment='top', transform=ax_list[4*plot+2].transAxes,bbox=dict(facecolor='none',alpha=1, edgecolor='none'))   
    
    ax_list[4*plot+2].plot(xc[start:stop],yc[start:stop],linewidth=0.5,color='black')    

    text=(str(steps) + ' steps')

    ax_list[4*plot+1].text(0,0,text, fontsize=12, horizontalalignment='left',
                       verticalalignment='top', transform=ax_list[4*plot+1].transAxes,bbox=dict(facecolor='none',alpha=1, edgecolor='none'))

    text=('N=' + str(steps) + '\n' +
          'p=' + str(p) + '\n' +
          '1/p=' + str(np.round(1/(p),3))     + '\n' +
          #str(np.array2string(np.arange(p+1)/p*2, formatter={'float_kind':lambda x: "%.1f" % x}, separator=', '))+ '$\pi$ \n\n' + 
          'X, Y $\in$ ' + str(np.array2string(np.arange(p)/(p-1)*2, formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())}, separator=', ')) +'\n\n' + 
          
          
          'measured:\n' + 
          'walked path=' + str(np.round(circum,3)) + '\n' +
          'diameter=' + str(np.round(maxy,3)) + '\n' +
          'circumference=' +  str(np.round(maxy*np.pi,3)))

    ax_list[4*plot+3].text(-0.12,0.98,text, fontsize=12, horizontalalignment='left', 
                       verticalalignment='top', transform=ax_list[4*plot+3].transAxes)

    text=('n='+ str(n) +'\n' +
          str(np.round(100*n/steps,4)) + '%'       
          ) 
    ax_list[4*plot+0].text(0.02,0.98,text, fontsize=12, horizontalalignment='left', 
                       verticalalignment='top', transform=ax_list[4*plot+0].transAxes)



def randomlist(p,steps):

    px=2*np.random.choice(p,steps)/(p-1)
    py=2*np.random.choice(p,steps)/(p-1)  
    return [px,py]

       

def videoloop(steps,parray):      
    d=1
    for q in range(105):
        p=2
        #if (q+3)%10==0:
        #    d=d*10
        #    continue  

        #n=(q+3)%10*d
        
        n=int((10**(q*0.06)))
        print(n)
        
    
        for i in range(5): 
            plotsetup(ax_list)
            plots(i,p,n,steps,parray)
            p=p+1
       
        plt.savefig('Random Walk' + str("%05d" % (q)), dpi=200, bbox_inches='tight')
        plotclear(ax_list)   

def single(n,steps,parray):
    p=2
    for i in range(5): 
        plots(i,p,n,steps,parray)
        plotsetup(ax_list)
        p=p+1
        plt.savefig('Random Walk' + str("%05d" % (111)), dpi=200, bbox_inches='tight')
    plt.show()

   

def main():
    n=10000
    steps=10000
    parray=[randomlist(2,steps),randomlist(3,steps),randomlist(4,steps),randomlist(5,steps),randomlist(6,steps)]
    #videoloop(steps,parray)
    single(n,steps,parray)  

main()     


 

