# load dependencies
library(randomForest)

set.seed(42)

# read data
train <- read.csv("../input/train.csv")
test <- read.csv("../input/test.csv")

# shuffle train
train <- train[sample(nrow(train)),]

# train model
rf <- randomForest(train[,3:34], train$Hazard, ntree=100, sampsize = 20000, imp=TRUE, do.trace=TRUE)

# make predition
submission <- data.frame(Id=test$Id)
submission$Hazard <- predict(rf, test[,2:33])
write.csv(submission, '../submissions/rf_model_2.csv', quote = FALSE, row.names = FALSE)
