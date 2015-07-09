# load dependencies
library(xgboost)
library(Matrix)
library(methods)

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
              "bst:max_depth" = 8,
              "min_child_weight" = 5,
              "subsample" = .8,
              "colsample_bytree" = .8,
              "scale_pos_weight" = 1)
offset = 5000
nround = 2500


y = log(y)
xgtrain = xgb.DMatrix(data = x[1:offset,], label=y[1:offset])
xgval = xgb.DMatrix(data = x[offset:nrow(train),], label=y[offset:nrow(train)])
watchlist = list(train = xgtrain, valid = xgval)

# Run Cross Valication
# bst.cv = xgb.cv(param=param, data = xgtrain, nfold = 3,
#                nrounds=nround, watchlist=watchlist, early.stop.round = 4,
#                 maximize = FALSE)
# [344]	train-rmse:2.933319+0.098488	test-rmse:3.847580+0.177589  ## ALL hot-ones

# Train the model
bst = xgb.train(param=param, data = xgtrain, nrounds=nround,
                watchlist=watchlist, maximize = FALSE, early.stop.round = 4)
# [1999]	train-rmse:1.436350	valid-rmse:3.914722 ## ALL hot-ones
# [1999]	train-rmse:2.688771	valid-rmse:2.700901 ## only 2 most important hot-ones

# Make prediction
pred = predict(bst,x[teind,])


testList = (offset+1):(offset*2)
xgtrain = xgb.DMatrix(data = x[testList,], label=y[testList])
xgval = xgb.DMatrix(data = x[(1:nrow(train))[-testList],], label=y[(1:nrow(train))[-testList]])
watchlist = list(train = xgtrain, valid = xgval)
bst = xgb.train(param=param, data = xgtrain, nrounds=nround,
                watchlist=watchlist, maximize = FALSE, early.stop.round = 4)

pred2 = predict(bst,x[teind,])

testList = (offset*2+1):(offset*3)
xgtrain = xgb.DMatrix(data = x[testList,], label=y[testList])
xgval = xgb.DMatrix(data = x[(1:nrow(train))[-testList],], label=y[(1:nrow(train))[-testList]])
watchlist = list(train = xgtrain, valid = xgval)
bst = xgb.train(param=param, data = xgtrain, nrounds=nround,
                watchlist=watchlist, maximize = FALSE, early.stop.round = 4)

pred3 = predict(bst,x[teind,])

testList = (offset*3+1):(offset*4)
xgtrain = xgb.DMatrix(data = x[testList,], label=y[testList])
xgval = xgb.DMatrix(data = x[(1:nrow(train))[-testList],], label=y[(1:nrow(train))[-testList]])
watchlist = list(train = xgtrain, valid = xgval)
bst = xgb.train(param=param, data = xgtrain, nrounds=nround,
                watchlist=watchlist, maximize = FALSE, early.stop.round = 4)

pred4 = predict(bst,x[teind,])

testList = (offset*4+1):(offset*5)
xgtrain = xgb.DMatrix(data = x[testList,], label=y[testList])
xgval = xgb.DMatrix(data = x[(1:nrow(train))[-testList],], label=y[(1:nrow(train))[-testList]])
watchlist = list(train = xgtrain, valid = xgval)
bst = xgb.train(param=param, data = xgtrain, nrounds=nround,
                watchlist=watchlist, maximize = FALSE, early.stop.round = 4)

pred5 = predict(bst,x[teind,])

testList = (offset*5+1):(offset*6)
xgtrain = xgb.DMatrix(data = x[testList,], label=y[testList])
xgval = xgb.DMatrix(data = x[(1:nrow(train))[-testList],], label=y[(1:nrow(train))[-testList]])
watchlist = list(train = xgtrain, valid = xgval)
bst = xgb.train(param=param, data = xgtrain, nrounds=nround,
                watchlist=watchlist, maximize = FALSE, early.stop.round = 4)

pred6 = predict(bst,x[teind,])

testList = (offset*6+1):(offset*7)
xgtrain = xgb.DMatrix(data = x[testList,], label=y[testList])
xgval = xgb.DMatrix(data = x[(1:nrow(train))[-testList],], label=y[(1:nrow(train))[-testList]])
watchlist = list(train = xgtrain, valid = xgval)
bst = xgb.train(param=param, data = xgtrain, nrounds=nround,
                watchlist=watchlist, maximize = FALSE, early.stop.round = 4)

pred7 = predict(bst,x[teind,])

testList = (offset*7+1):(offset*8)
xgtrain = xgb.DMatrix(data = x[testList,], label=y[testList])
xgval = xgb.DMatrix(data = x[(1:nrow(train))[-testList],], label=y[(1:nrow(train))[-testList]])
watchlist = list(train = xgtrain, valid = xgval)
bst = xgb.train(param=param, data = xgtrain, nrounds=nround,
                watchlist=watchlist, maximize = FALSE, early.stop.round = 4)

pred8 = predict(bst,x[teind,])

testList = (offset*8+1):(offset*9)
xgtrain = xgb.DMatrix(data = x[testList,], label=y[testList])
xgval = xgb.DMatrix(data = x[(1:nrow(train))[-testList],], label=y[(1:nrow(train))[-testList]])
watchlist = list(train = xgtrain, valid = xgval)
bst = xgb.train(param=param, data = xgtrain, nrounds=nround,
                watchlist=watchlist, maximize = FALSE, early.stop.round = 4)

pred9 = predict(bst,x[teind,])

testList = (offset*9+1):nrow(train)
xgtrain = xgb.DMatrix(data = x[testList,], label=y[testList])
xgval = xgb.DMatrix(data = x[(1:nrow(train))[-testList],], label=y[(1:nrow(train))[-testList]])
watchlist = list(train = xgtrain, valid = xgval)
bst = xgb.train(param=param, data = xgtrain, nrounds=nround,
                watchlist=watchlist, maximize = FALSE, early.stop.round = 4)

pred10 = predict(bst,x[teind,])


predTotal = pred + pred2 + pred3 + pred4 + pred5 + pred6 + pred7 + pred8 + pred9 + pred10

# Output submission
predTotal = format(predTotal, digits=8,scientific=F)
submission = data.frame(Id = test$Id,Hazard=predTotal)
write.csv(submission,file='../submissions/xgboost_model_8.csv', quote=FALSE,row.names=FALSE)
