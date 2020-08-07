rm(list=ls())

library(tidyverse)
library(corrplot)
library(caret)
library(lares)

# Get/Read the data

citation <- "Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6."
citation2 <- "Read more: http://groupware.les.inf.puc-rio.br/har#ixzz6U6EfKn6V"
citation3 <- "Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013."

learnFile <- "training.csv"
learnUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(learnUrl,learnFile)
learn <- as_tibble(read.csv(learnFile))

evaluateFile <- "testing.csv"
evaluateUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(evaluateUrl,evaluateFile)
evaluate <- as_tibble(read.csv(evaluateFile))

# Find columns named differently in train and test set and make them the same
difNames <- which(names(evaluate)!=names(learn))
print(paste("Element(s) #s:", difNames,"are different"))
names(evaluate)[difNames] <- names(learn)[difNames]

# Make classe a factor
learn$classe <- factor(learn$classe)
evaluate$classe <- factor(evaluate$classe)

# Eliminate the first 7 columns of both datasets
learn <- learn %>% select(-(1:7))
evaluate <- evaluate %>% select(-(1:7))

# Find columns in the evaluate dataset that contain only NA and remove them
naColumns <- names(which(apply(evaluate,2, function(x) all(is.na(x)))))
useColumns <- names(evaluate)[!(names(evaluate) %in% naColumns)]

evaluate <- evaluate[,useColumns]
learn <- learn[,useColumns]

# Check that no NA's are left in either dataset
if(length(names(which(apply(evaluate,2, function(x) any(is.na(x)))))=="")==0) print("No NAs in evaluate")
if(length(names(which(apply(learn,2, function(x) any(is.na(x)))))=="")==0) print("No NAs in learn")

# Split the learn dataset into a training, a testing and a validation datasets
set.seed(1234)
inBuild <- createDataPartition(learn$classe, p = 3/4)[[1]]
validation <- learn[-inBuild,]
buildData <- learn[inBuild,]

inTrain = createDataPartition(buildData$classe, p = 3/4)[[1]]
training = buildData[ inTrain,]
testing = buildData[-inTrain,]

# Are there columns with near-zero variance in training?  If yes, remove them
nearZeroVariance <- nearZeroVar(training)
if(length(nearZeroVariance)==0) {
        print("No near-zero-variance columns in training")
} else {
        print("Removing near-zero-variance columns:")
        print(nearZeroVariance)
        training <- training[,-nearZeroVariance]
        testing <- testing[,-nearZeroVariance]
        validation <- validation[,-nearZeroVar]
        evaluate <- evaluate[,-nearZeroVariance]
}

# Are there highly correlated columns?  If yes, remove them
corrM <- cor(select(training, -classe))
diag(corrM)<-0
corrplot(corrM, method="circle",tl.cex=0.6)
highCorr <- findCorrelation(corrM, cutoff = 0.9)
if(length(highCorr)==0) {
        print("No highly correlated columns in training")
} else {
        print( paste("Removing", length(highCorr), "highly correlated columns:"))
        print(names(training)[highCorr])
        training <- training[,-highCorr]
        testing <- testing[,-highCorr]
        validation <- validation[,-highCorr]
        evaluate <- evaluate[,-highCorr]
}

# Build five models
trainctrl <- trainControl(verboseIter = TRUE)

modelRF <- train(classe~., method="rf", data=training, trControl=trainctrl) #Random Forest
modelGBM <- train(classe~., method="gbm", data=training, verbose=FALSE, trControl=trainctrl) #Gradient Boosting Machine
modelRPART <- train(classe~., method="rpart", data=training, trControl=trainctrl) #CART
modelTreebag <- train(classe~., method="treebag", data=training, trControl=trainctrl) #Bagged CART
modelLDA <- train(classe~., method="lda", data=training, trControl=trainctrl) #Linear Discriminant Analysis

# Evaluate the models

predRF <- predict(modelRF, testing)
predGBM <- predict(modelGBM, testing)
predRPART <- predict(modelRPART, testing)
predTreebag <- predict(modelTreebag, testing)
predLDA <- predict(modelLDA, testing)

cmRF <- confusionMatrix(predRF,testing$classe)
cmGBM <- confusionMatrix(predGBM,testing$classe)
cmRPART <- confusionMatrix(predRPART,testing$classe)
cmTreebag <- confusionMatrix(predTreebag,testing$classe)
cmLDA <- confusionMatrix(predLDA,testing$classe)

cmRF$overall[1]
cmGBM$overall[1]
cmRPART$overall[1]
cmTreebag$overall[1]
cmLDA$overall[1]

# Build combined model using models with accuracy > 0.8

predDataFrame <- data.frame(predRF, 
                            predGBM, 
                            predTreebag, 
                            classe=testing$classe)
modelCombined <- train(classe~.,model="rf", data=predDataFrame,trControl=trainctrl)
predCombined <- predict(modelCombined, predDataFrame)

cmCombined <- confusionMatrix(predCombined,testing$classe)

cmCombined$overall[1]

# Evaluate on the validation dataset

vpredRF <- predict(modelRF, validation)
vpredGBM <- predict(modelGBM, validation)
vpredTreebag <- predict(modelTreebag, validation)

vcmRF <- confusionMatrix(vpredRF,validation$classe)
vcmGBM <- confusionMatrix(vpredGBM,validation$classe)
vcmTreebag <- confusionMatrix(vpredTreebag,validation$classe)

vcmRF$overall[1]
vcmGBM$overall[1]
vcmTreebag$overall[1]


vpredDataFrame <- data.frame(predRF=vpredRF,
                             predGBM=vpredGBM,
                             predTreebag=vpredTreebag)
vpredCombined <- predict(modelCombined,vpredDataFrame)

vcmCombined <- confusionMatrix(vpredCombined,validation$classe)
vcmCombined$overall[1]

# Evaluate predictions for unknown dataset

epredRF <- predict(modelRF, evaluate)
epredGBM <- predict(modelGBM, evaluate)
epredTreebag <- predict(modelTreebag, evaluate)

epredDataFrame <- data.frame(predRF=epredRF,
                             predGBM=epredGBM,
                             predTreebag=epredTreebag)
epredCombined <- predict(modelCombined,epredDataFrame)

ePredictions <- data.frame(RF=epredRF,
                           GBM=epredGBM,
                           Treebag=epredTreebag,
                           Combined=epredCombined)


# Look at the best models 


plot(varImp(modelRF))
plot(varImp(modelTreeBag))

qplot(yaw_belt,pitch_forearm,colour=classe,data=training)
qplot(yaw_belt,pitch_belt,colour=classe,data=training)

probRF <- predict(modelRF,testing,type="prob")



tag <- testing$classe
score <- predRF
mplot_full(tag,score, multis=probRF)
mplot_density(tag,score)
mplot_roc(tag,score,multis=probRF)
mplot_response(tag, score, multis=probRF)

tag <- testing$classe
score <- predRF
mplot_full(tag,score, multis=probRF)
mplot_density(tag,score)
mplot_roc(tag,score,multis=probRF)

#plot(modelGBM)

#plot(modelRPART)
#plot(varImp(modelRPART))

