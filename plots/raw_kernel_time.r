#!/usr/bin/Rscript

require(lattice);
require(reshape2);
require(ggplot2);

a<-read.csv('raw_kernel_time.csv',header=T);
pdf(file="raw_kernel_time.pdf", height=3.5, width=7.5);

b<-data.frame(operation=a$operation,model=a$model,time=a$runtime);
c<-aggregate(b[3],b[1:2],min);
d<-melt(c,id.vars=c("operation","model"));
d$operation=d$operation+1;
#d=d[-c(16),]
 
p <- ggplot(data=d,aes(x=operation,y=value,group=model,fill=factor(model)))+geom_bar(width=0.5,stat="identity",position=position_dodge()) +
scale_x_discrete(breaks=seq(0,6,1),labels=c('','add','sub','mult','div','exp','log'))+
scale_fill_manual(labels=c('range','error','area'),values=c('#40E0D0','#FF6347',"#C7C7C7"))+
xlab('Arithmetic Operation') + ylab('Time per thread (ms)');

p + theme_bw() + theme(legend.position=c(0.85,0.8), legend.title=element_blank(),legend.background = element_blank(),legend.text=element_text(size=10)); 

