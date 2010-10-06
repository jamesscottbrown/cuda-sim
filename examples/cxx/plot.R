
plot_immdeath <- function(){
	times <- c(0:40)
	d1 <- read.table('res_ssa_immdeath.txt')
	d2 <- read.table('res_sde_immdeath.txt')  

	d1 <- d1[,-1]
	d2 <- d2[,-1]

	matplot( times, t(as.matrix(d1)), type='l', col=1 )
	matlines( times, t(as.matrix(d2)), type='l', col=2 )
	legend('bottom',legend=c('ssa','sde'), lty=1, col=c(1,2))
}

plot_p53 <- function(){

	times <- c(0:40)
	d1 <- read.table('res_ssa_p53.txt')
	d2 <- read.table('res_sde_p53.txt')  

	d1 <- d1[,-1]
	d2 <- d2[,-1]

	d1s1 <- d1[ seq(1,nrow(d1),by=3), ]
	d2s1 <- d2[ seq(1,nrow(d1),by=3), ]

	matplot( times, t(as.matrix(d1s1)), type='l', col=1 )
	matlines( times, t(as.matrix(d2s1)), type='l', col=2 )
	legend('bottom',legend=c('ssa','sde'), lty=1, col=c(1,2))
}