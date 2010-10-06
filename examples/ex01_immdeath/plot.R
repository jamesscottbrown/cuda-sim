
read.table('results/immdeath_MJP_result.txt',header=F) -> a
#read.table('results/immdeath_SDE_result.txt',header=F) -> a
#read.table('results/immdeath_ODE_result.txt',header=F) -> a

cols = length(a[2,])
maxVal = 0

for (i in c(1:10)){
    maxVal = max(maxVal,max(a[i+1,4:cols]))
} 


#pdf('result.pdf', height=3, width=6)
x11(height=3, width=6)

#set margins
# south, west, north, east
par(mar=c(4.5, 4.2, 2, 1.5))

plot(x=t(a[1,4:cols]),y=t(a[2,4:cols]),main="Results",ylim=c(0,maxVal), xlab="time", ylab="count", type="l")
for (i in c(1:9)){
    points(x=t(a[1,4:cols]),y=t(a[i+2,4:cols]),col = i+1,type="l")
}

#dev.off()
