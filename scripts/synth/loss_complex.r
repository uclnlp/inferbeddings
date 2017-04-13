x = seq(-1,1,0.01)
fs = 1.8
fsc = 1.8


#complEx simple box, on differences
pdf("complex_simple_box.pdf",width=7,height=7)
op <- par(mar = c(5,5.5,2,2) +0.1)
contour(x,x,outer(x,x, function(x,y) pmax(0,x) + pmax(x,abs(y))),nlevels=10,cex.lab=fs+0.4, cex.axis=fs, cex.main=fs, cex.sub=fs, labcex=fsc, xlab=expression(Re(theta['b, i']- theta['r, i'])), ylab=expression(Im(theta['b, i'] - theta['r, i'])))
dev.off()


#complex simple sphere
pdf("complex_simple_sphere.pdf",width=7,height=7)
op <- par(mar = c(5,5.5,2,2) +0.1)
contour(x,x,outer(x,x, function(x,y) sqrt(x*x + y*y)),nlevels=10,cex.lab=fs+0.4, cex.axis=fs, cex.main=fs, cex.sub=fs, labcex=fsc, xlab=expression(Re(theta['b, i'] - theta['r, i'])), ylab=expression(Im(theta['b, i'] - theta['r, i'])))
dev.off()

