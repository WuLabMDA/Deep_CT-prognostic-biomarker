# set directory
setwd("C:/Users/MBSaad/Desktop/GitHub Version/Sub-network4")
# delete work space
rm(list = ls(all = TRUE))
graphics.off()

library("R.matlab")
library(ConsensusClusterPlus)
library("survival")
library("survcomp")
library(SNFtool)
library(survminer)
library(cluster)
library(openxlsx)

#******************Training**********************

# read train data
modelname = "train_features"
tmp <- read.csv(paste(modelname,".csv",sep=""))
train_data <- tmp[,-1]

# read train_events
modelname = "train_events"
tmp <- read.csv(paste(modelname,".csv",sep=""))
clinic.train <- tmp[,-1]

trainDist <- as.matrix(daisy(train_data, metric = "gower"))

# cluster data
rtrain = ConsensusClusterPlus(
  trainDist,
  maxK=4,
  reps=100,
  distance="spearman",
  tmyPal=c("white","seagreen"),
  clusterAlg="pam")

subgroup <- 2

label.train <- unlist(rtrain[[subgroup]]["consensusClass"])
label.train[label.train==1] = "S1"
label.train[label.train==2] = "S2"

clinic.train$label <- label.train
train.fit <- survfit(Surv(OS_time/12,OS_status) ~ label, data = clinic.train)

custom_theme <- function() {
  theme_survminer() %+replace%
    theme(
      plot.title=element_text(size = 14, color = "black",hjust=0.5,face = "bold"),
      axis.text.x = element_text(size = 14, color = "black", face = "bold"),
      legend.text = element_text(size = 14, color = "black", face = "bold"),
      legend.title = element_text(size = 14, color = "black", face = "bold"),
      axis.text.y = element_text(size = 14, color = "black", face = "bold"),
      axis.title.x = element_text(size = 14, color = "black", face = "bold"),
      axis.title.y = element_text(size = 14, color = "black", face = "bold") , #angle=(90))
    )
}

ggsurvplot(train.fit, data = clinic.train,title = "Training",ggtheme=custom_theme(),
           conf.int = TRUE,
           pval = TRUE,
           fun = "pct",
           risk.table = TRUE,
           xlab = "Survival time (Months)",
           ylab = "Survival (%)",
           xlim = c(0, 5.5),
           risk.table.fontsize =5,
           size = 1,
           linetype = "strata",
           palette = c("#E7B800",
                       "#2E9FDF"),
           
           risk.table.col = "strata",
           #legend = "bottom",
           legend.title = "Group",
           legend.labs = c("S1",
                           "S2"))


#******************Validation**********************

# read valid data
modelname = "valid_features"
tmp <- read.csv(paste(modelname,".csv",sep=""))
valid_data <- tmp[,-1]

# read valid_events
modelname = "valid_events"
tmp <- read.csv(paste(modelname,".csv",sep=""))
clinic.valid <- tmp[,-1]

validDist <- as.matrix(daisy(valid_data, metric = "gower"))

# cluster data
rvalid = ConsensusClusterPlus(
  validDist,
  maxK=4,
  reps=100,
  distance="spearman",
  tmyPal=c("white","seagreen"),
  clusterAlg="pam")

subgroup <- 2

label.valid <- unlist(rvalid[[subgroup]]["consensusClass"])
label.valid[label.valid==1] = "S1"
label.valid[label.valid==2] = "S2"

clinic.valid$label <- label.valid
valid.fit <- survfit(Surv(OS_time/12,OS_status) ~ label, data = clinic.valid)

ggsurvplot(valid.fit, data = clinic.valid,title = "Validation",ggtheme=custom_theme(),
           conf.int = TRUE,
           pval = TRUE,
           fun = "pct",
           risk.table = TRUE,
           xlab = "Survival time (Months)",
           ylab = "Survival (%)",
           xlim = c(0, 5.5),
           risk.table.fontsize =5,
           size = 1,
           linetype = "strata",
           palette = c("#E7B800",
                       "#2E9FDF"),
           
           risk.table.col = "strata",
           #legend = "bottom",
           legend.title = "Group",
           legend.labs = c("S1",
                           "S2"))