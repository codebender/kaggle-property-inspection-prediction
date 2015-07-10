# load dependencies
library(xgboost)
library(Matrix)
library(methods)

# set seed
set.seed(42)

# read data
train <- read.csv("../input/train.csv", stringsAsFactors = FALSE)
test <- read.csv("../input/test.csv", stringsAsFactors = FALSE)

### drop id cols
train = train[,-1]

### shuffle
train <- train[sample(nrow(train)),]

y = train$Hazard

allData = rbind(train[,-1],test[,-1])

### convert categorical columns
allData$T1_V6[allData$T1_V6 == "Y"] = 1
allData$T1_V6[allData$T1_V6 == "N"]  = 0
allData$T1_V17[allData$T1_V17 == "Y"] = 1
allData$T1_V17[allData$T1_V17 == "N"]  = 0
allData$T2_V3[allData$T2_V3 == "Y"] = 1
allData$T2_V3[allData$T2_V3 == "N"]  = 0
allData$T2_V11[allData$T2_V11 == "Y"] = 1
allData$T2_V11[allData$T2_V11 == "N"]  = 0
allData$T2_V12[allData$T2_V12 == "Y"] = 1
allData$T2_V12[allData$T2_V12 == "N"]  = 0

allData <- cbind(allData, with(allData, data.frame(model.matrix(~T1_V4-1,allData))))
allData$T1_V4 <- NULL

allData <- cbind(allData, with(allData, data.frame(model.matrix(~T1_V5-1,allData))))
allData$T1_V5 <- NULL

allData <- cbind(allData, with(allData, data.frame(model.matrix(~T1_V7-1,allData))))
allData$T1_V7 <- NULL

allData <- cbind(allData, with(allData, data.frame(model.matrix(~T1_V8-1,allData))))
allData$T1_V8 <- NULL

allData <- cbind(allData, with(allData, data.frame(model.matrix(~T1_V9-1,allData))))
allData$T1_V9 <- NULL

allData <- cbind(allData, with(allData, data.frame(model.matrix(~T1_V11-1,allData))))
allData$T1_V11 <- NULL

allData <- cbind(allData, with(allData, data.frame(model.matrix(~T1_V12-1,allData))))
allData$T1_V12 <- NULL

allData <- cbind(allData, with(allData, data.frame(model.matrix(~T1_V15-1,allData))))
allData$T1_V15 <- NULL

allData <- cbind(allData, with(allData, data.frame(model.matrix(~T1_V16-1,allData))))
allData$T1_V16 <- NULL

allData <- cbind(allData, with(allData, data.frame(model.matrix(~T2_V5-1,allData))))
allData$T2_V5 <- NULL

allData <- cbind(allData, with(allData, data.frame(model.matrix(~T2_V13-1,allData))))
allData$T2_V13 <- NULL


### Convert to numeric matrix
x = as.matrix(allData)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

# Set necessary parameters
param <- list("objective" = "reg:linear",
              "nthread" = 8,
              "bst:eta" = .01,
              "bst:max_depth" = 7,
              "min_child_weight" = 5,
              "subsample" = .8,
              "colsample_bytree" = .85,
              "scale_pos_weight" = 1)
offset = 5000
nround = 2000
early_stopping = 6

y = log(y)
xgtrain = xgb.DMatrix(data = x[1:offset,], label=y[1:offset])
xgval = xgb.DMatrix(data = x[offset:nrow(train),], label=y[offset:nrow(train)])
watchlist = list(valid = xgval, train = xgtrain)

# Train the model
bst = xgb.train(param=param, data = xgtrain, nrounds=nround,
                watchlist=watchlist, maximize = FALSE,
                early.stop.round = early_stopping)
# [331]	valid-rmse:0.861011	train-rmse:0.713695
pred = predict(bst,x[teind,])


testList = (offset+1):(offset*2)
xgtrain = xgb.DMatrix(data = x[testList,], label=y[testList])
xgval = xgb.DMatrix(data = x[(1:nrow(train))[-testList],], label=y[(1:nrow(train))[-testList]])
watchlist = list(valid = xgval, train = xgtrain)
bst = xgb.train(param=param, data = xgtrain, nrounds=nround,
                watchlist=watchlist, maximize = FALSE,
                early.stop.round = early_stopping)
# [347]	valid-rmse:0.860506	train-rmse:0.696842
pred2 = predict(bst,x[teind,])

testList = (offset*2+1):(offset*3)
xgtrain = xgb.DMatrix(data = x[testList,], label=y[testList])
xgval = xgb.DMatrix(data = x[(1:nrow(train))[-testList],], label=y[(1:nrow(train))[-testList]])
watchlist = list(valid = xgval, train = xgtrain)
bst = xgb.train(param=param, data = xgtrain, nrounds=nround,
                watchlist=watchlist, maximize = FALSE,
                early.stop.round = early_stopping)
# [358]	valid-rmse:0.859786	train-rmse:0.709583
pred3 = predict(bst,x[teind,])

testList = (offset*3+1):(offset*4)
xgtrain = xgb.DMatrix(data = x[testList,], label=y[testList])
xgval = xgb.DMatrix(data = x[(1:nrow(train))[-testList],], label=y[(1:nrow(train))[-testList]])
watchlist = list(valid = xgval, train = xgtrain)
bst = xgb.train(param=param, data = xgtrain, nrounds=nround,
                watchlist=watchlist, maximize = FALSE,
                early.stop.round = early_stopping)
# [313]	valid-rmse:0.861349	train-rmse:0.714383
pred4 = predict(bst,x[teind,])

testList = (offset*4+1):(offset*5)
xgtrain = xgb.DMatrix(data = x[testList,], label=y[testList])
xgval = xgb.DMatrix(data = x[(1:nrow(train))[-testList],], label=y[(1:nrow(train))[-testList]])
watchlist = list(valid = xgval, train = xgtrain)
bst = xgb.train(param=param, data = xgtrain, nrounds=nround,
                watchlist=watchlist, maximize = FALSE,
                early.stop.round = early_stopping)
# [362]	valid-rmse:0.861249	train-rmse:0.708691
pred5 = predict(bst,x[teind,])

testList = (offset*5+1):(offset*6)
xgtrain = xgb.DMatrix(data = x[testList,], label=y[testList])
xgval = xgb.DMatrix(data = x[(1:nrow(train))[-testList],], label=y[(1:nrow(train))[-testList]])
watchlist = list(valid = xgval, train = xgtrain)
bst = xgb.train(param=param, data = xgtrain, nrounds=nround,
                watchlist=watchlist, maximize = FALSE,
                early.stop.round = early_stopping)
# [407]	valid-rmse:0.860177	train-rmse:0.685815
pred6 = predict(bst,x[teind,])

testList = (offset*6+1):(offset*7)
xgtrain = xgb.DMatrix(data = x[testList,], label=y[testList])
xgval = xgb.DMatrix(data = x[(1:nrow(train))[-testList],], label=y[(1:nrow(train))[-testList]])
watchlist = list(valid = xgval, train = xgtrain)
bst = xgb.train(param=param, data = xgtrain, nrounds=nround,
                watchlist=watchlist, maximize = FALSE,
                early.stop.round = early_stopping)
# [304]	valid-rmse:0.860340	train-rmse:0.720759
pred7 = predict(bst,x[teind,])

testList = (offset*7+1):(offset*8)
xgtrain = xgb.DMatrix(data = x[testList,], label=y[testList])
xgval = xgb.DMatrix(data = x[(1:nrow(train))[-testList],], label=y[(1:nrow(train))[-testList]])
watchlist = list(valid = xgval, train = xgtrain)
bst = xgb.train(param=param, data = xgtrain, nrounds=nround,
                watchlist=watchlist, maximize = FALSE,
                early.stop.round = early_stopping)
# [383]	valid-rmse:0.859623	train-rmse:0.710071
pred8 = predict(bst,x[teind,])

testList = (offset*8+1):(offset*9)
xgtrain = xgb.DMatrix(data = x[testList,], label=y[testList])
xgval = xgb.DMatrix(data = x[(1:nrow(train))[-testList],], label=y[(1:nrow(train))[-testList]])
watchlist = list(valid = xgval, train = xgtrain)
bst = xgb.train(param=param, data = xgtrain, nrounds=nround,
                watchlist=watchlist, maximize = FALSE,
                early.stop.round = early_stopping)
# [355]	valid-rmse:0.863137	train-rmse:0.701130
pred9 = predict(bst,x[teind,])

testList = (offset*9+1):nrow(train)
xgtrain = xgb.DMatrix(data = x[testList,], label=y[testList])
xgval = xgb.DMatrix(data = x[(1:nrow(train))[-testList],], label=y[(1:nrow(train))[-testList]])
watchlist = list(valid = xgval, train = xgtrain)
bst = xgb.train(param=param, data = xgtrain, nrounds=nround,
                watchlist=watchlist, maximize = FALSE,
                early.stop.round = early_stopping)
# [361]	valid-rmse:0.859115	train-rmse:0.719561
pred10 = predict(bst,x[teind,])


predTotal = pred + pred2 + pred3 + pred4 + pred5 + pred6 + pred7 + pred8 + pred9 + pred10

# Output submission
predTotal = format(predTotal, digits=8,scientific=F)
submission = data.frame(Id = test$Id,Hazard=predTotal)
write.csv(submission,file='../submissions/xgboost_model_17.csv', quote=FALSE,row.names=FALSE)
