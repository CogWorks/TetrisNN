require(ggplot2)
require(reshape2)
while (T) {
  d <- read.csv("~/workspace/TetrisNN/deepqlearning.log",header=F)
  colnames(d) <- c("game","episodes","lines","score","reward")
  d <- melt(d,id.vars="game")
  p <- ggplot(d,aes(x=game,y=value,group=variable,color=variable)) + geom_line() + stat_smooth(color="black",size=1.25)
  print(p)
  Sys.sleep(10)
}