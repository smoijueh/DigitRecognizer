# Samuel Moijueh

# Load Packages
# vis
library(knitr)    # dynamic report generation
library(ggplot2)  # visualization
library(scales)   # scale functions for visualization
library(gridExtra) # multi-panel plots

# wrangle
library(EBImage)  # image manipulation/processing
library(reshape2) # data frame manipulation
library(kableExtra) # HTML table manipulation
library(magrittr) # pipe commands

get_KD_tree_prediction_labels<-function(kd_tree, trainData){
  
  get_k_nearest_votes<-function(i,trainData,kd_tree){
    k_nearest_votes <- trainData[c(kd_tree[i,]), 1]
    tt <- table(k_nearest_votes)
    most_common_label <- names(tt[which.max(tt)])
    return(most_common_label)
  }
  
  count <- nrow(kd_tree)
  
  labeled_predictions <- NA
  
  for (i in 1:count){
    labeled_predictions[i] <- get_k_nearest_votes(i,trainData,kd_tree)
  }
  labeled_predictions <- factor(labeled_predictions, levels = 0:9)
  
  return(labeled_predictions)
}


# Final Model
train <- read.csv("../input/train.csv")
train$label <- as.factor(train$label)
test <- read.csv("../input/test.csv")

trainData <- removeEmptyPixels(train)

# Compute the principal components
pca.result <- princomp(trainData[,-1], scores = T)
train.pca <- pca.result$scores
test.pca <- predict(pca.result, test)

# number of principal components
num.principal.components <- 1:49
# train model
kd_treeModel <- nabor::knn(data = train.pca[,num.principal.components], query = test.pca[,num.principal.components], k=4)

# get predictions
Label <- get_KD_tree_prediction_labels(kd_treeModel$nn.idx,train)

# write predictions to file
ImageId = c(1:length(Label))
submission <- data.frame(ImageId,Label)
write.csv(submission, file = "../mnist_kdtree_kNN_submission.csv", row.names = F)
