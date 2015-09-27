# Self movement prediction
Adrián Álvarez del Castillo  
September 25, 2015  
***
```
This is a R Markdown document.
```
# Introduction  
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.  

# Data preprocessing
First step is to load the requiered packages.

```r
library(knitr)
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(rpart)
library(rpart.plot)
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(corrplot)
```



**Download the Data**  
The information provided for the analysis is dowloaded from the web.

```r
# Define URL and file names
trainUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainFile <- "./data/pml-training.csv"
testFile  <- "./data/pml-testing.csv"

# Check existance of the local target directory
if (!file.exists("./data")) {dir.create("./data")}

# Download files
if (!file.exists(trainFile))
        {download.file(trainUrl, destfile=trainFile, method="curl")}
if (!file.exists(testFile))
        {download.file(testUrl, destfile=testFile, method="curl")}
```

**Read the Data**  
Once the data has been downloaded, it is posible to read it into from the csv files into a couple of data frames.

```r
trainRaw <- read.csv("./data/pml-training.csv")
testRaw <- read.csv("./data/pml-testing.csv")
```
The training data set contains 19,622 observations and 160 variables, while the testing data set contains 20 observations and 160 variables. The "classe" variable in the training set is the outcome to predict. 

```r
barplot(table(trainRaw$classe), xlab = "Classe", col = "steelblue")
```

<img src="Figs/unnamed-chunk-5-1.png" title="" alt="" style="display: block; margin: auto;" />

**Clean the data**  
In this step the data will be cleaned by removing the variables with exclusively missing values.

```r
trainCleaned <- trainRaw[, colSums(is.na(trainRaw)) == 0] 
testCleaned <- testRaw[, colSums(is.na(testRaw)) == 0] 
```
Next, all non-numeric variables and variables related to timestamps will be removed.

```r
classe <- trainCleaned$classe
trainRemove <- grepl("^X|timestamp|window", names(trainCleaned))
trainCleaned <- trainCleaned[, !trainRemove]
trainCleaned <- trainCleaned[, sapply(trainCleaned, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(testCleaned))
testCleaned <- testCleaned[, !testRemove]
testCleaned <- testCleaned[, sapply(testCleaned, is.numeric)]
```
After the data cleansing, the training data set contains 19,622 observations and 53 variables. The target variable (Classe) is retained in the cleaned train data set.

**Slice the data**  
For model training purposes the cleaned data set will be partitioned in two, one for training with 60% of observations and one for validation with 40% of observations. The validation data set will be use to conduct cross validation.  

```r
set.seed(1970) # For reproducibile purpose
inTrain <- createDataPartition(trainCleaned$classe, p=0.60, list=F)
trainData <- trainCleaned[inTrain, ]
testData <- trainCleaned[-inTrain, ]
```
# Modeling
For predictive activity recognition a model will be fitted using **Random Forest** algorithm.  

Random Forests are a combination of tree predictors where each tree depends on the values of a random vector sampled independently with the same distribution for all trees in the forest. The basic principle is that a group of “weak learners” can come together to form a “strong learner”. Random Forests are a great tool for making predictions considering they do not overfit because of the law of large numbers. Introducing the right kind of randomness makes them accurate classifiers and regressors.  

When applying the algorithm **5-fold cross-validation** will be use.

Cross-validation, sometimes called rotation estimation, is a model validation technique for assessing how the results of a statistical analysis will generalize to an independent data set. It is mainly used in settings where the goal is prediction, and one wants to estimate how accurately a predictive model will perform in practice.

```r
controlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=trainData, method="rf", trControl=controlRf, ntree=250)
modelRf
```

```
## Random Forest 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 9421, 9420, 9421, 9421, 9421 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9872627  0.9838843  0.003029274  0.003836065
##   27    0.9893003  0.9864639  0.002886258  0.003651271
##   52    0.9843749  0.9802344  0.004836416  0.006116813
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```
The performance of the model on the validation data set is estimated with the predict function.

```r
predictRf <- predict(modelRf, testData)
confusionMatrix(testData$classe, predictRf)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2229    2    0    0    1
##          B   11 1496   11    0    0
##          C    0    8 1352    8    0
##          D    0    2   21 1262    1
##          E    0    0    1    4 1437
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9911         
##                  95% CI : (0.9887, 0.993)
##     No Information Rate : 0.2855         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9887         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9951   0.9920   0.9762   0.9906   0.9986
## Specificity            0.9995   0.9965   0.9975   0.9963   0.9992
## Pos Pred Value         0.9987   0.9855   0.9883   0.9813   0.9965
## Neg Pred Value         0.9980   0.9981   0.9949   0.9982   0.9997
## Prevalence             0.2855   0.1922   0.1765   0.1624   0.1834
## Detection Rate         0.2841   0.1907   0.1723   0.1608   0.1832
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9973   0.9943   0.9868   0.9935   0.9989
```


```r
accuracy <- postResample(predictRf, testData$classe)
accuracy
```

```
##  Accuracy     Kappa 
## 0.9910783 0.9887138
```

```r
ose <- 1 - as.numeric(confusionMatrix(testData$classe, predictRf)$overall[1])
```
The estimated out-of-sample error is 0.892%.

# Predicting test cases
The fitted model is applied to the provided testing cases obtained from the data source.

```r
result <- predict(modelRf, testCleaned[, -length(names(testCleaned))])
result
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
The required files are generated for later submission.

```r
pml_write_files <- function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_results/problem_id_",i,".txt")
                write.table(x[i], file=filename, quote=FALSE,
                            row.names=FALSE, col.names=FALSE)
                }
        }
pml_write_files(result)
```

# Appendix: Figures
**Correlation Matrix Visualization**

```r
corrPlot <- cor(trainData[, -length(names(trainData))])
corrplot(corrPlot, method="color")
```

<img src="Figs/unnamed-chunk-14-1.png" title="" alt="" style="display: block; margin: auto;" />

**Variable importance**

```r
plot(varImp(modelRf, scale=F), cex=1.5, col="steelblue")
```

<img src="Figs/unnamed-chunk-15-1.png" title="" alt="" style="display: block; margin: auto;" />

**Decision Tree Visualization**

```r
treeModel <- rpart(classe ~ ., data=trainData, method="class")
prp(treeModel)
```

<img src="Figs/unnamed-chunk-16-1.png" title="" alt="" style="display: block; margin: auto;" />
