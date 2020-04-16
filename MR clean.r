
### Simulation DGP for month
### I'm still refining this but the basic structure is good
### Generate firm-characteristics and returns of two models (linear model and nonlinear model) and two cases (P=100 and P=200)   

path='./Simu/'
name1='SimuData_100'
name2='SimuData_200'
dir.create(path)
dir.create(paste(path,name1,sep=''))
dir.create(paste(path,name2,sep=''))

### Locally-Run Codes MCMC=100

for(M in 1:100){


set.seed(M*123)

### 200 Variables Case ###
n=200
m=100
T=180
stdv=0.05
stde=0.05


# Generate Characteristics
rho=runif(m,0.9,1)
c=matrix(0,n*T,m)
for(i in 1:m){
  x=matrix(0,n,T)
  x[,1]=rnorm(n)
  for(t in 2:T){
    x[,t]=rho[i]*x[,t-1]+rnorm(n)*sqrt(1-rho[i]^2)
  }
  x1=apply(x,2,rank)*2/(n+1)-1
  c[,i]=as.vector(x1)
}


# Generate Factors
per=rep(1:n,T)
time=rep(1:T,each=n)

vt=matrix(rnorm(T*3),3,T)*stdv
beta=c[,c(1,2,3)]
betav=numeric(n*T)
for(t in 1:T){
  ind=time==t
  betav[ind]=beta[ind,]%*%vt[,t]
}

# Generate Macro TS variable
y=numeric(T)
y[1]=rnorm(1)
q=0.95
for(t in 2:T){
  y[t]=q*y[t-1]+rnorm(1)*sqrt(1-q^2)
}

cy=c
for(t in 1:T){
  ind=time==t
  cy[ind,]=c[ind,]*y[t]
}
ep=rt(n*T,5)*stde



### Model 1 (Linear Model)

theta_w=0.02
theta=c(c(1,1),rep(0,m-2),0,0,1,rep(0,m-3))*theta_w
r1=cbind(c,cy)%*%theta+betav+ep
write.csv(cbind(c,cy),paste(path,name2,'/c',M,'.csv',sep=''),row.names = F) 
write.csv(r1,paste(path,name2,'/r1_',M,'.csv',sep=''),row.names = F)


### Model 2 (Nonlinear Model)

theta=c(c(1,1),rep(0,m-2),0,0,1,rep(0,m-3))*theta_w
z=cbind(c,cy)
z[,1]=c[,1]^2*2
z[,2]=c[,1]*c[,2]*1.5
z[,m+3]=sign(cy[,3])*0.6
r2=z%*%theta+betav+ep
write.csv(r2,paste(path,name2,'/r2_',M,'.csv',sep=''),row.names = F)



### 100 Variables Case ###

m=50
### Model 1 (Linear Model)
theta=c(c(1,1),rep(0,m-2),0,0,1,rep(0,m-3))*theta_w
r1=cbind(c[,1:m],cy[,1:m])%*%theta+betav+ep
write.csv(cbind(c[,1:m],cy[,1:m]),paste(path,name1,'/c',M,'.csv',sep=''),row.names = F) 
write.csv(r1,paste(path,name1,'/r1_',M,'.csv',sep=''),row.names = F)


### Model 2 (Nonlinear Model)
theta=c(c(1,1),rep(0,m-2),0,0,1,rep(0,m-3))*theta_w
z=cbind(c[,1:m],cy[,1:m])
z[,1]=c[,1]^2*2
z[,2]=c[,1]*c[,2]*1.5
z[,m+3]=sign(cy[,3])*0.6
r2=z%*%theta+betav+ep
write.csv(r2,paste(path,name1,'/r2_',M,'.csv',sep=''),row.names = F)

}
