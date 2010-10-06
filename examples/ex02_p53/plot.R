read.table('results/p53_MJP_result.txt',header=F) -> a
#read.table('results/p53_SDE_result.txt',header=F) -> a
#read.table('results/p53_ODE_result.txt',header=F) -> a

cols = length(a[2,])

maxVal = max(a[2,4:cols],a[3,4:cols],a[4,4:cols],
a[5,4:cols],a[6,4:cols],a[7,4:cols])


#pdf('result.pdf', height=6, width=6)
x11(height=6, width=6)
par(mfrow=c(2,1))

#set margins
# south, west, north, east
par(mar=c(4.5, 4.2, 2, 1.5))

plot(x=t(a[1,4:cols]),y=t(a[2,4:cols]),main="Result 1",ylim=c(0,maxVal), xlab="time", ylab="count", type="l")
for (i in c(1:2)){
    points(x=t(a[1,4:cols]),y=t(a[i+2,4:cols]),col = i+1,type="l")
}

plot(x=t(a[1,4:cols]),y=t(a[5,4:cols]),main="Result 2",ylim=c(0,maxVal), xlab="time", ylab="count", type="l")
for (i in c(1:2)){
    points(x=t(a[1,4:cols]),y=t(a[i+5,4:cols]),col = i+1,type="l")
}

#dev.off()