library(dplyr)
library(tidyverse)
library(caret)
library(MASS)
library(pROC)

train.raw<-read.csv("train.csv")
test.raw<-read.csv("test.csv")

nrow(train.raw)# 891
#  check for missing values: 
sapply(train.raw,function(x) sum(is.na(x))) # there are 177 subjects with missing age 
summary(train.raw)
# Factorize categorical variables
train.clean <- train.raw %>%
  mutate(
    Survived = factor(Survived, levels = c(0, 1), labels = c("No", "Yes")),
    Pclass = factor(Pclass, levels = c(1:3), labels = c("1st", "2nd", "3rd")),
    Name = factor(Name),
    Sex = factor(Sex),
    Cabin = factor(Cabin),
    Embarked = factor(ifelse(Embarked=="","S",Embarked))
  ) %>%
  dplyr::select(Survived, Pclass, Name, Sex, Age, SibSp, Parch, Fare, Cabin, Embarked)
summary(train.clean)
test.clean <- test.raw %>%
  mutate(
    Pclass = factor(Pclass, levels = c(1:3), labels = c("1st", "2nd", "3rd")),
    Name = factor(Name),
    Sex = factor(Sex),
    Cabin = factor(Cabin),
    Embarked = factor(ifelse(Embarked=="","S",Embarked))
  ) %>%
  dplyr::select( Pclass, Name, Sex, Age, SibSp, Parch, Fare, Cabin, Embarked)

summary(test.clean)
# Replace missing Age values based on title in Name
train.clean$Age <- ifelse(
  is.na(train.clean$Age),
  ifelse(
    grepl("Mrs\\.", train.clean$Name), 
    mean(train.clean[grepl("Mrs\\.", train.clean$Name), ]$Age, na.rm = TRUE),
    ifelse(
      grepl("Mr\\.", train.clean$Name), 
      mean(train.clean[grepl("Mr\\.", train.clean$Name), ]$Age, na.rm = TRUE),
      ifelse(
        grepl("Miss\\.", train.clean$Name), 
        mean(train.clean[grepl("Miss\\.", train.clean$Name), ]$Age, na.rm = TRUE),
        ifelse(
          grepl("Master\\.", train.clean$Name), 
          mean(train.clean[grepl("Master\\.", train.clean$Name), ]$Age, na.rm = TRUE),
          ifelse(
            grepl("Dr\\.", train.clean$Name), 
            mean(train.clean[grepl("Dr\\.", train.clean$Name), ]$Age, na.rm = TRUE),
            NA
          )
        )
      )
    )
  ),
  train.clean$Age
)

test.clean$Age <- ifelse(
  is.na(test.clean$Age),
  ifelse(
    grepl("Mrs\\.", test.clean$Name), 
    mean(test.clean[grepl("Mrs\\.", test.clean$Name), ]$Age, na.rm = TRUE),
    ifelse(
      grepl("Mr\\.", test.clean$Name), 
      mean(test.clean[grepl("Mr\\.", test.clean$Name), ]$Age, na.rm = TRUE),
      ifelse(
        grepl("Miss\\.", test.clean$Name), 
        mean(test.clean[grepl("Miss\\.", test.clean$Name), ]$Age, na.rm = TRUE),
        ifelse(
          grepl("Master\\.", test.clean$Name), 
          mean(test.clean[grepl("Master\\.", test.clean$Name), ]$Age, na.rm = TRUE),
          ifelse(
            grepl("Dr\\.", test.clean$Name), 
            mean(test.clean[grepl("Dr\\.", test.clean$Name), ]$Age, na.rm = TRUE),
            median(test.clean$Age,na.rm=T)
          )
        )
      )
    )
  ),
  test.clean$Age
)
summary(test.clean)

library(DescTools)
# fill missing 'Embarked' with mode(train set)
#train.clean$Embarked[train.clean$Embarked == ""] <- as.character(Mode(train.clean$Embarked))
# fill missing 'Fare' with median (test set)
test.clean[is.na(test.clean$Fare),]$Fare <- median(test.clean$Fare,na.rm = T)

# Separate predictors and response
train.clean.x <- train.clean %>%
  dplyr::select(Age, Pclass, Sex, SibSp, Parch, Fare, Embarked)
train.clean.y <- train.clean %>%
  dplyr::select(Survived)

summary(train.clean)
# Set up cross-validation parameters
set.seed(123)
ctrl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

# Define model training with cross-validation
train_models <- function(train.data, train.labels) {
  
  # Logistic Regression
  log.cv <- train(
    x = train.clean.x,  # Use full training data
    y = train.clean.y$Survived,
    method = "glm",
    trControl = ctrl,
    metric = "ROC"
  )
  
  # Random Forest
  rf.cv <- train(
    x = train.clean.x,  # Use full training data
    y = train.clean.y$Survived,
    method = "rf",
    trControl = ctrl,
    metric = "ROC",
    tuneGrid = expand.grid(
      mtry = seq(2, ncol(train.clean.x), 1)
    )
  )
  
  # Neural Network
  nnet_grid <- expand.grid(
    size = c(5, 10, 15),
    decay = c(0.001, 0.01, 0.1)
  )
  
  nn.cv <- train(
    x = train.clean.x,  # Use full training data
    y = train.clean.y$Survived,
    method = "nnet",
    trControl = ctrl,
    metric = "ROC",
    tuneGrid = nnet_grid,
    trace = FALSE,
    maxit = 1000,
    linout = FALSE
  )
  
  return(list(
    logistic = log.cv,
    randomforest = rf.cv,
    neuralnet = nn.cv
  ))
}
# Train models using full training data
models <- train_models(train.clean.x, train.clean.y)

# Extract and compare performance metrics from cross-validation
cv_performance <- function(models) {
  performance_summary <- data.frame(
    Model = names(models),
    ROC = sapply(models, function(x) max(x$results$ROC))
  )
  return(performance_summary)
}
cv_performance(models) # random forest seems to be the best 
summary(train.clean.x)
test.clean.x<-test.clean%>%dplyr::select(Age,Pclass,Sex,SibSp,Parch,Fare,Embarked)
# Make predictions on test set
test_predictions <- predict(models$randomforest,test.clean.x,"raw")

test.raw$test_predictions<-test_predictions
write.csv(test.raw,"pred.csv")