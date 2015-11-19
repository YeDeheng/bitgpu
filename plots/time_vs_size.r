#!/usr/bin/Rscript

require(scales);
require(ggplot2);
require(reshape2);

pdf(file="time_vs_size.pdf", height=3.5, width=5)

ref<-read.csv('../data/sizes.dat',header=F,check.names=F)

n=12
df <- data.frame(var = numeric(n), time = numeric(n), dataset = character(n), stringsAsFactors = FALSE, check.names=F);

add_entry <- function(df, i, var, a, name)
{
	a_t=max(a[,2])
	df$var[i] <- var
	df$time[i] <- a_t
	df$dataset[i] <- as.character(name)
	return (df)
}

a<-read.csv('../data/raw_asa_poly.dat',header=F)
df <- add_entry (df,1,ref[1,2],a,ref[1,1]);
c<-read.csv('../data/raw_bitslice_poly.dat',header=F)
df <- add_entry (df,2,ref[1,2],c,ref[1,1]);

a0<-read.csv('../data/raw_asa_poly6.dat',header=F)
df <- add_entry (df,3,ref[2,2],a0,ref[2,1]);
c0<-read.csv('../data/raw_bitslice_poly6.dat',header=F)
df <- add_entry (df,4,ref[2,2],c0,ref[2,1]);

#a1a<-read.csv('../data/raw_asa_poly8.dat',header=F)
#df <- add_entry (df,5,ref[3,2],a1a,ref[3,1]);
#c1a<-read.csv('../data/raw_bitslice_poly8.dat',header=F)
#df <- add_entry (df,6,ref[3,2],c1a,ref[3,1]);

a1<-read.csv('../data/raw_asa_diode.dat',header=F)
df <- add_entry (df,5,ref[4,2],a1,ref[4,1]);
c1<-read.csv('../data/raw_bitslice_diode.dat',header=F)
df <- add_entry (df,6,ref[4,2],c1,ref[4,1]);

a2<-read.csv('../data/raw_asa_level1_linear.dat',header=F)
df <- add_entry (df,7,ref[5,2],a2,ref[5,1]);
c2<-read.csv('../data/raw_bitslice_level1_linear.dat',header=F)
df <- add_entry (df,8,ref[5,2],c2,ref[5,1]);

a3<-read.csv('../data/raw_asa_level1_satur.dat',header=F)
df <- add_entry (df,9,ref[6,2],a3,ref[6,1]);
c3<-read.csv('../data/raw_bitslice_level1_satur.dat',header=F)
df <- add_entry (df,10,ref[6,2],c3,ref[6,1]);

a4<-read.csv('../data/raw_asa_approx1.dat',header=F)
df <- add_entry (df,11,ref[7,2],a4,ref[7,1]);
c4<-read.csv('../data/raw_bitslice_approx1.dat',header=F)
df <- add_entry (df,12,ref[7,2],c4,ref[7,1]);

# odd/even extraction
df2<-data.frame(df[seq(1,12,2),],check.names=F);
df3<-data.frame(df[seq(2,12,2),],check.names=F)

p <- ggplot() +
	geom_line(data=df2,aes(x=var,y=time,col="CPU"),size=1.1,linetype="solid") +
	geom_line(data=df3,aes(x=var,y=time,col="GPU",linetype="dashed"),size=1.1,linetype="dashed") +
	geom_point(data=df,aes(x=var,y=time,shape=dataset),size=3)+
	scale_x_continuous(limits=c(5,20),breaks=pretty_breaks()) + 
	scale_y_log10() +
	ylab("Time (s)") + xlab("#Variables") +
	theme_bw();

#	scale_x_log10(limits=c(1,100)) + 
	
p + theme(legend.key=element_blank(),legend.title=element_blank(),legend.background = element_blank(),legend.text=element_text(size=10));
