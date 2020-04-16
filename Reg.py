
### Simulation Regression Models
### All Regressions 
### OLS, OLS+H, PCR, PLS, Lasso, Lasso+H, Ridge, Ridge+H, ENet, ENet+H and Group Lasso, Group Lasso+H.
### Including the Oracel Regression Model  

### Server-Run Codes (Run 1 MCMC Simu on each node)
import argparse
args = argparse.ArgumentParser()
args.add_argument("Symbol", help="MCMC")
arg = args.parse_args()
number = arg.Symbol
MC=int(number) 



import numpy as np
import pandas as pd
from scipy import linalg, optimize
from sklearn import linear_model
import os
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
import random 
random.seed(MC*123)
import timeit
start = timeit.default_timer()


### run P=100 and P=200 Cases seperately
datanum='100'   
#datanum='200'
path='/project2/dachxiu/gushihao/research/Simu'
dirstock=path+'/SimuData_'+datanum+'/'


### Fuctions ###
def fw2(x):
    d=x.shape
    m=np.max(x)
    for i in range(d[0]):
        for j in range(d[1]):
            if x[i,j]==m:
                return [i,j]

def fw1(x):
    d=len(x)
    m=np.max(x)
    for i in range(d):
        if x[i]==m:
            return i
def sq(a,b,step):
    r=[]
    new=a
    r=r+[a]
    for i in range(100000):
        new=new+step
        if new<=b:
            r=r+[new]
        else:
            break
    return r

def cut_knots_degree2(x,n,th):
    a=x.shape[0]
    if len(x.shape)==1:
        b=1
    else:
        b=x.shape[1]
    
    resultfinal=np.zeros((a,b*(n+1)))
    for i in range(b):
        xcut=x[:,i]
        xcutnona=np.copy(xcut)
        xcutnona[np.isnan(xcutnona)]=0
        index=(1-1*np.isnan(xcut))==1

        t=th[:,i]

        x1=np.copy(xcutnona)
        resultfinal[:,(n+1)*i]=x1-np.mean(x1)
        x1=np.power(np.copy(xcutnona)-t[0],2)
        resultfinal[:,(n+1)*i+1]=x1-np.mean(x1)
       
        for j in range(1,n):
            x1=np.power(xcutnona-t[j],2)*(xcutnona>=t[j])
            resultfinal[:,(n+1)*i+1+j]=x1-np.mean(x1)
    return resultfinal

def loss(y,yhat):
    return np.mean(np.power(yhat-y,2))
def lossh(y,yhat,mu):
    r=abs(yhat-y)
    l=np.zeros(len(r))
    ind=r>mu
    l[ind]=2*mu*r[ind]-mu*mu
    ind=r<=mu
    l[ind]=r[ind]*r[ind]
    return np.mean(l)

def f_grad(XX,XY,w):
    return  XX.dot(w)-XY
def f_gradh(w,X,y,mu):
    r=np.squeeze(np.asarray(X.dot(w)-y))
    g=np.zeros(len(w))
    N=len(r)
    p=len(w)

    for i in range(N):
        if r[i]>mu:
            g=g+mu*X[i,:]
        elif r[i]<-mu:
            g=g-mu*X[i,:]
        else:
            g=g+r[i]*X[i,:]
    return g.reshape(p,1)

def soft_threshodl(w,mu):
    return np.multiply(np.sign(w), np.maximum(np.abs(w)- mu,0))  
def soft_threshodr(w,mu):
    return w/(1+mu)
def soft_threshode(w,mu):
    return np.multiply(np.sign(w), np.maximum(np.abs(w)- 0.5*mu,0)) /(1+0.5*mu)
def soft_threshoda(w,alpha,mu):
    return np.multiply(np.sign(w), np.maximum(np.abs(w)- alpha*mu,0)) /(1+alpha*mu)
def soft_threshodg(w,mu):
    w1=np.copy(w)
    for i in range(nc):
        ind=groups==i
        wg=w1[ind,:]
        nn=wg.shape[0]
        n2=np.sqrt(np.sum(np.power(wg,2)))
        if n2<=mu:
            w1[ind,:]=np.zeros((nn,1))
        else:
            w1[ind,:]=wg-mu*wg/n2
    return w1

def proximal(XX,XY,tol,L,l1,func):
    dim = XX.shape[0]
    max_iter = 30000
    gamma = 1/L
    w = np.matrix([0.0]*dim).T
    v = w
    for t in range(0, max_iter):
        vold=np.copy(v)
        w_prev = w
        w = v - gamma * f_grad(XX,XY,v)
        w = func(w,l1*gamma)
        v = w + t/(t+3) * (w - w_prev)
        if np.sum(np.power(v-vold,2))< (np.sum(np.power(vold,2))*tol) or np.sum(np.abs(v-vold))==0:
            break
    return np.squeeze(np.asarray(v))

def proximalH(w,X,y,mu,tol,L,l1,func):
    
    max_iter = 30000
    gamma = 1/L
    v = w

    for t in range(0, max_iter):
        vold=np.copy(v)
        w_prev = w
        w = v - gamma * f_gradh(v,X,y,mu)
        w = func(w,l1*gamma)
        v = w + t/(t+3) * (w - w_prev)
        if np.sum(np.power(v-vold,2))< (np.sum(np.power(vold,2))*tol) or np.sum(np.abs(v-vold))==0:
            break
        
    return np.squeeze(np.asarray(v))

def PCR(X,y,A):
    
    XX=X.T.dot(X)
    pca=np.linalg.eig(XX)
    p1=pca[1][:,:A]
    Z=X.dot(p1)

    B=np.zeros((X.shape[1],A))
    for i in range(A-1):
        xx=Z[:,:(i+1)]
        b=np.linalg.pinv(xx.T.dot(xx)).dot(xx.T).dot(y)
        b=p1[:,:(i+1)].dot(b)
        B[:,i+1]=b
    return B

def pls(X,y,A):
    # X is convariates matrix (N x p), y is response vector (should be demeaned),  A is the number of steps.
    # return coefficients matrix (p x A)
    s=X.T.dot(y)
    R=np.zeros((X.shape[1],A))
    TT=np.zeros((X.shape[0],A))
    P=np.zeros((X.shape[1],A))
    U=np.zeros((X.shape[0],A))
    V=np.zeros((X.shape[1],A))
    Q=np.zeros((1,A))
    B=np.zeros((X.shape[1],A))
    
    for i in range(A):
        q=s.T.dot(s)
        r=s*q
        t=X.dot(r)
        t=t-np.mean(t)
        normt=np.sqrt(t.T.dot(t))
        t=t/normt
        r=r/normt
        p=X.T.dot(t)
        q=y.T.dot(t)
        u=y*q
        v=np.copy(p)
        if i>0:
            v=v-V[:,:i].dot(V[:,:i].T.dot(p))
            u=u-TT[:,:i].dot(TT[:,:i].T.dot(u))
        v=v/np.sqrt(v.T.dot(v))
        s=s-v.dot(v.T.dot(s))
        
        R[:,i]=r
        TT[:,i]=t
        P[:,i]=p
        U[:,i]=u
        V[:,i]=v
        Q[:,i]=q
    
    for i in range(A-1):
        B[:,i+1]=R[:,:(i+1)].dot(Q[:,:(i+1)].T)[:,0]
    return  B

def vip(b,xtrain,ytrain,mtrain):
    yhatbig1=xtrain.dot(b)+mtrain
    r2=1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))
    v=np.zeros(len(b))
    for i in range(len(b)):
        b1=np.copy(b)
        b1[i]=0
        yhatbig1=xtrain.dot(b1)+mtrain
        r2new=1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))
        v[i]=r2-r2new
    return v



for hh in [1,3,6,12]:
    title=path+'/Simu_'+datanum+'/Reg%d'%hh
    if not os.path.exists(title) and MC==1:
        os.makedirs(title)
    if not os.path.exists(title+'/B') and MC==1:
        os.makedirs(title+'/B')
    if not os.path.exists(title+'/VIP') and MC==1:
        os.makedirs(title+'/VIP')

### hh is the horizon parameter, e.g. hh=1 means using monthly return as response variable. hh=3 is quarterly, hh=6 is Half-year and hh=12 is annually.
for hh in [1,3,6,12]:

    title=path+'/Simu_'+datanum+'/Reg%d'%hh
    if datanum=='100':
        nump=50
    if datanum=='200':
        nump=100

    mu=0.2*np.sqrt(hh)
    tol=1e-10

    
    ### Start to MCMC ###

    for M in [MC]:
        for mo in [1,2]:
            
            N=200   ### Number of CS tickers
            m=nump*2   ### Number of Characteristics
            T=180   ### Number of Time Periods

            per=np.tile(np.arange(N)+1,T)
            time=np.repeat(np.arange(T)+1,N)
            stdv=0.05
            theta_w=0.005

            c=pd.read_csv(dirstock+'c%d.csv'%M,delimiter=',').values
            r1=pd.read_csv(dirstock+'r%d_%d_%d.csv'%(mo,M,hh),delimiter=',').iloc[:,0].values

            ### Add Some Elements ###
            daylen=np.repeat(N,T/3)
            daylen_test=daylen
            ind=range(0,(N*T/3))
            xtrain=c[ind,:]
            ytrain=r1[ind]
            trainper=per[ind]
            ind=range((N*T/3),(N*(T*2/3-hh+1)))
            xtest=c[ind,:]
            ytest=r1[ind]
            testper=per[ind]

            l1=c.shape[0]
            l2=len(r1)
            l3=l2-np.sum(np.isnan(r1))
            print l1,l2,l3
            ind=range((N*T*2/3),min(l1,l2,l3))
            xoos=c[ind,:]
            yoos=r1[ind]
            del c
            del r1

            ### Demean Returns ### 
            ytrain_demean=ytrain-np.mean(ytrain)
            ytest_demean=ytest-np.mean(ytest)
            mtrain=np.mean(ytrain)
            mtest=np.mean(ytest)


            ### Calculate Sufficient Stats ###
            sd=np.zeros(xtrain.shape[1])
            for i in range(xtrain.shape[1]):
                s=np.std(xtrain[:,i])
                if s>0:
                    xtrain[:,i]=xtrain[:,i]/s
                    xtest[:,i]=xtest[:,i]/s
                    xoos[:,i]=xoos[:,i]/s
                    sd[i]=s

            XX=xtrain.T.dot(xtrain)
            b=np.linalg.svd(XX)
            L=b[1][0]
            print 'Lasso L=',L
            Y=np.matrix(ytrain_demean).T
            XY=xtrain.T.dot(Y)


            ### Start to Train ###
            
            r2_oos=np.zeros(13)  ### OOS R2
            r2_is=np.zeros(13)   ### IS R2
            

            ### OLS ###
            
            modeln=0
            clf=linear_model.LinearRegression(fit_intercept=False, normalize=False)
            clf.fit(xtrain,ytrain_demean)
            yhatbig1=clf.predict(xoos)+mtrain
            r2_oos[modeln]=1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))
            yhatbig1=clf.predict(xtrain)+mtrain
            r2_is[modeln]=1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))

            b=clf.coef_
            df=pd.DataFrame(b)
            df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
            v=vip(b,xtrain,ytrain,mtrain)
            df=pd.DataFrame(v)
            df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
            
            print '###Simple OLS OK!###'

            modeln+=1
            b=proximalH(np.matrix(b).T,xtrain,np.matrix(ytrain_demean).T,mu,tol,L,0,soft_threshodl)

            yhatbig1=xoos.dot(b)+mtrain
            r2_oos[modeln]=1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))
            yhatbig1=xtrain.dot(b)+mtrain
            r2_is[modeln]=1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))
            df=pd.DataFrame(b)
            df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
            v=vip(b,xtrain,ytrain,mtrain)
            df=pd.DataFrame(v)
            df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)

            print '###Simple OLS+H OK!###'



            ### PCA Regression ###
            ### Tuning parameter: the number of PCs

            modeln+=1
            ne=30
            B=PCR(xtrain,ytrain_demean,ne)
            r=np.zeros((3,ne))

            for j in range(ne):
                
                b=B[:,j]
                yhatbig1=xtest.dot(b)+mtrain
                r[0,j]=1-sum(np.power(yhatbig1-ytest,2))/sum(np.power(ytest-mtrain,2))
                yhatbig1=xoos.dot(b)+mtrain
                r[1,j]=1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))
                yhatbig1=xtrain.dot(b)+mtrain
                r[2,j]=1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))

            r2_oos[modeln]=r[1,int(fw1(r[0,:]))]    
            r2_is[modeln]=r[2,int(fw1(r[0,:]))]    
            b=B[:,int(fw1(r[0,:]))]
            df=pd.DataFrame(b)
            df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
            v=vip(b,xtrain,ytrain,mtrain)
            df=pd.DataFrame(v)
            df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)

            print '###PCA Regression Good!###'

            ### PLS Regression ###
            ### Tuning parameter: the number of components

            modeln+=1
            ne=30
            B=pls(xtrain,ytrain_demean,ne)
            r=np.zeros((3,ne))

            for j in range(ne):
                
                b=B[:,j]
                yhatbig1=xtest.dot(b)+mtrain
                r[0,j]=1-sum(np.power(yhatbig1-ytest,2))/sum(np.power(ytest-mtrain,2))
                yhatbig1=xoos.dot(b)+mtrain
                r[1,j]=1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))
                yhatbig1=xtrain.dot(b)+mtrain
                r[2,j]=1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))

            r2_oos[modeln]=r[1,int(fw1(r[0,:]))]    
            r2_is[modeln]=r[2,int(fw1(r[0,:]))]    
            b=B[:,int(fw1(r[0,:]))]
            df=pd.DataFrame(b)
            df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
            v=vip(b,xtrain,ytrain,mtrain)
            df=pd.DataFrame(v)
            df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)

            print '###PLS Regression Good!###'


        
            ### Lasso ###
            ### Tuning parameter: the L1 penalty lambda

            modeln+=1
            lamv=sq(-2,4,0.1)
            alpha=1
            r=np.zeros((3,len(lamv)))

            for j in range(len(lamv)):
                l2=10**lamv[j]
                b=proximal(XX,XY,tol,L,l2,soft_threshodl)
                yhatbig1=xtest.dot(b)+mtrain
                r[0,j]=1-sum(np.power(yhatbig1-ytest,2))/sum(np.power(ytest-mtrain,2))
                yhatbig1=xoos.dot(b)+mtrain
                r[1,j]=1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))
                yhatbig1=xtrain.dot(b)+mtrain
                r[2,j]=1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))

            r2_oos[modeln]=r[1,int(fw1(r[0,:]))]    
            r2_is[modeln]=r[2,int(fw1(r[0,:]))]    
            l2=10**lamv[int(fw1(r[0,:]))]
            print 'Lasso',l2,'[-2,4]'
            b=proximal(XX,XY,tol,L,l2,soft_threshodl)
            df=pd.DataFrame(b)
            df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
            v=vip(b,xtrain,ytrain,mtrain)
            df=pd.DataFrame(v)
            df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
            
            print '###Lasso Good!###'
            
            modeln+=1
            b=proximalH(np.matrix(b).T,xtrain,np.matrix(ytrain_demean).T,mu,tol,L,l2,soft_threshodl)

            yhatbig1=xoos.dot(b)+mtrain
            r2_oos[modeln]=1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))
            yhatbig1=xtrain.dot(b)+mtrain
            r2_is[modeln]=1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))
            df=pd.DataFrame(b)
            df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
            v=vip(b,xtrain,ytrain,mtrain)
            df=pd.DataFrame(v)
            df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
            print '###Lasso+H Good!###'




            ### Ridge ### 
            ### Tuning parameter: the L2 penalty lambda
            modeln+=1
            lamv=sq(0,6,0.1)
            alpha=1
            r=np.zeros((3,len(lamv)))

            for j in range(len(lamv)):
                l2=10**lamv[j]
                b=proximal(XX,XY,tol,L,l2,soft_threshodr)
                yhatbig1=xtest.dot(b)+mtrain
                r[0,j]=1-sum(np.power(yhatbig1-ytest,2))/sum(np.power(ytest-mtrain,2))
                yhatbig1=xoos.dot(b)+mtrain
                r[1,j]=1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))
                yhatbig1=xtrain.dot(b)+mtrain
                r[2,j]=1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))

            r2_oos[modeln]=r[1,int(fw1(r[0,:]))]    
            r2_is[modeln]=r[2,int(fw1(r[0,:]))]    
            l2=10**lamv[int(fw1(r[0,:]))]
            print 'Ridge',l2,'[0,6]'
            b=proximal(XX,XY,tol,L,l2,soft_threshodr)
            df=pd.DataFrame(b)
            df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
            v=vip(b,xtrain,ytrain,mtrain)
            df=pd.DataFrame(v)
            df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
            
            print '###Ridge Good!###'
            
            modeln+=1
            b=proximalH(np.matrix(b).T,xtrain,np.matrix(ytrain_demean).T,mu,tol,L,l2,soft_threshodr)

            yhatbig1=xoos.dot(b)+mtrain
            r2_oos[modeln]=1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))
            yhatbig1=xtrain.dot(b)+mtrain
            r2_is[modeln]=1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))
            df=pd.DataFrame(b)
            df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
            v=vip(b,xtrain,ytrain,mtrain)
            df=pd.DataFrame(v)
            df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
            print '###Ridge+H Good!###'




            ### Elastic Net ###
            ### Tuning parameter: the L1+L2 penalty lambda
            modeln+=1
            lamv=sq(-2,4,0.1)
            alpha=0.5
            r=np.zeros((3,len(lamv)))


            for j in range(len(lamv)):
                l2=10**lamv[j]
                b=proximal(XX,XY,tol,L,l2,soft_threshode)
                yhatbig1=xtest.dot(b)+mtrain
                r[0,j]=1-sum(np.power(yhatbig1-ytest,2))/sum(np.power(ytest-mtrain,2))
                yhatbig1=xoos.dot(b)+mtrain
                r[1,j]=1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))
                yhatbig1=xtrain.dot(b)+mtrain
                r[2,j]=1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))

            r2_oos[modeln]=r[1,int(fw1(r[0,:]))]    
            r2_is[modeln]=r[2,int(fw1(r[0,:]))]    
            l2=10**lamv[int(fw1(r[0,:]))]
            print 'Enet',l2,'[-2,4]'
            b=proximal(XX,XY,tol,L,l2,soft_threshode)
            df=pd.DataFrame(b)
            df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
            v=vip(b,xtrain,ytrain,mtrain)
            df=pd.DataFrame(v)
            df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
            
            print '###Enet Good!###'
            
            modeln+=1
            b=proximalH(np.matrix(b).T,xtrain,np.matrix(ytrain_demean).T,mu,tol,L,l2,soft_threshode)

            yhatbig1=xoos.dot(b)+mtrain
            r2_oos[modeln]=1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))
            yhatbig1=xtrain.dot(b)+mtrain
            r2_is[modeln]=1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))
            df=pd.DataFrame(b)
            df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
            v=vip(b,xtrain,ytrain,mtrain)
            df=pd.DataFrame(v)
            df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
            print '###Enet+H Good!###'




            ### Oracle Models ###
            modeln+=1
            if mo==1:
                x=np.zeros((xtrain.shape[0],3))
                x[:,0]=xtrain[:,0]
                x[:,1]=xtrain[:,1]
                x[:,2]=xtrain[:,nump+2]
                x1=np.zeros((xoos.shape[0],3))
                x1[:,0]=xoos[:,0]
                x1[:,1]=xoos[:,1]
                x1[:,2]=xoos[:,nump+2]

                clf=linear_model.LinearRegression(fit_intercept=False, normalize=False)
                clf.fit(x,ytrain)
                yhatbig1=clf.predict(x1)
                r2_oos[modeln]=1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))

                yhatbig1=clf.predict(x)
                r2_is[modeln]=1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))
                print '###Oracle OLS!###'



            if mo==2:
                x=np.zeros((xtrain.shape[0],3))
                x[:,0]=np.power(xtrain[:,0],2)
                x[:,1]=xtrain[:,1]*xtrain[:,0]
                x[:,2]=np.sign(xtrain[:,nump+2])
                x1=np.zeros((xoos.shape[0],3))
                x1[:,0]=np.power(xoos[:,0],2)
                x1[:,1]=xoos[:,1]*xoos[:,0]
                x1[:,2]=np.sign(xoos[:,nump+2])

                clf=linear_model.LinearRegression(fit_intercept=False, normalize=False)
                clf.fit(x,ytrain)
                yhatbig1=clf.predict(x1)
                r2_oos[modeln]=1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))

                yhatbig1=clf.predict(x)
                r2_is[modeln]=1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))
                print '###Oracle OLS!###'      



            ### Group Lasso ###
            ### Tuning parameter: the group lasso penalty lambda

            kn=4 # the number of knots
            th=np.zeros((kn,xtrain.shape[1]))
            th[2,:]=0
            for i in range(xtrain.shape[1]):
                th[:,i]=np.percentile(xtrain[:,i],np.arange(kn)*100.0/kn)

            xtrain=cut_knots_degree2(xtrain,kn,th)
            xtest=cut_knots_degree2(xtest,kn,th)
            xoos=cut_knots_degree2(xoos,kn,th)
            
            for i in range(xtrain.shape[1]):
                s=np.std(xtrain[:,i])
                if s>0:
                    xtrain[:,i]=xtrain[:,i]/s
                    xtest[:,i]=xtest[:,i]/s
                    xoos[:,i]=xoos[:,i]/s

            Y=np.matrix(ytrain_demean).T
            XX=xtrain.T.dot(xtrain)
            b=np.linalg.svd(XX)
            print 'L=',b[1][0]
            L=b[1][0]
            XY=xtrain.T.dot(Y)

            modeln+=1
            lamv=sq(0.5,3,0.1)
            nc=(XX.shape[1])/(kn+1)
            groups=np.repeat(range(nc),kn+1)
            r=np.zeros((3,len(lamv)))

            for j in range(len(lamv)):
                l2=10**lamv[j]
                b=proximal(XX,XY,tol,L,l2,soft_threshodg)
                yhatbig1=xtest.dot(b)+mtrain
                r[0,j]=1-sum(np.power(yhatbig1-ytest,2))/sum(np.power(ytest-mtrain,2))
                yhatbig1=xoos.dot(b)+mtrain
                r[1,j]=1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))
                yhatbig1=xtrain.dot(b)+mtrain
                r[2,j]=1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))
         

            r2_oos[modeln]=r[1,int(fw1(r[0,:]))]    
            r2_is[modeln]=r[2,int(fw1(r[0,:]))]    
            l2=10**lamv[int(fw1(r[0,:]))]
            print 'GLasso',l2,'[0.5,3]'
            b=proximal(XX,XY,tol,L,l2,soft_threshodg)
            df=pd.DataFrame(b)
            df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)

            v=vip(b,xtrain,ytrain,mtrain)
            df=pd.DataFrame(v)
            df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
            
            print '###Group Lasso Good!###'
            
            modeln+=1
            b=proximalH(np.matrix(b).T,xtrain,np.matrix(ytrain_demean).T,mu,tol,L,l2,soft_threshodg)

            yhatbig1=xoos.dot(b)+mtrain
            r2_oos[modeln]=1-sum(np.power(yhatbig1-yoos,2))/sum(np.power(yoos-mtrain,2))
            yhatbig1=xtrain.dot(b)+mtrain
            r2_is[modeln]=1-sum(np.power(yhatbig1-ytrain,2))/sum(np.power(ytrain-mtrain,2))
            df=pd.DataFrame(b)
            df.to_csv(title+'/B/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)

            v=vip(b,xtrain,ytrain,mtrain)
            df=pd.DataFrame(v)
            df.to_csv(title+'/VIP/b%d_%d_%d.csv'%(mo,M,modeln),header=False, index=False)
            
            print '###Group Lasso+H Good!###'




            print r2_oos
            df=pd.DataFrame(r2_oos)
            df.to_csv(title+'/roos_%d_%d.csv'%(mo,M),header=False, index=False)

            print r2_is
            df=pd.DataFrame(r2_is)
            df.to_csv(title+'/ris_%d_%d.csv'%(mo,M),header=False, index=False)


stop = timeit.default_timer()
print('Time: ', stop - start)

