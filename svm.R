library(DMwR)
library(e1071)
library(h2o)
library(caret)
library(lme4)
library(ggalluvial)
library(xgboost)
library(jsonlite)
library(lubridate)
library(knitr)
library(Rmisc)
library(scales)
library(countrycode)
library(highcharter)
library(glmnet)
library(keras)
library(forecast)
library(zoo)
library(magrittr)
library(tidyverse)

set.seed(30)

setwd("/Users/p0d00cn/Documents/Learning/Google RP/all/")

tr <- read_csv("train.csv")
te <- read_csv("test.csv")
subm <- read_csv("sample_submission.csv")

## JSON data
flatten_json <- . %>% 
  str_c(., collapse = ",") %>% 
  str_c("[", ., "]") %>% 
  fromJSON(flatten = T)

parse <- . %>% 
  bind_cols(flatten_json(.$device)) %>%
  bind_cols(flatten_json(.$geoNetwork)) %>% 
  bind_cols(flatten_json(.$trafficSource)) %>% 
  bind_cols(flatten_json(.$totals)) %>% 
  select(-device, -geoNetwork, -trafficSource, -totals)
tr <- parse(tr)
te <- parse(te)

## Train and test features sets intersection
setdiff(names(tr), names(te))
tr %<>% select(-one_of("campaignCode"))

## Constant columns
fea_uniq_values <- sapply(tr, n_distinct)
(fea_del <- names(fea_uniq_values[fea_uniq_values == 1]))

tr %<>% select(-one_of(fea_del))
te %<>% select(-one_of(fea_del))

is_na_val <- function(x) x %in% c("not available in demo dataset", "(not provided)",
                                  "(not set)", " ", "unknown.unknown",  "(none)")

tr %<>% mutate_all(funs(ifelse(is_na_val(.), NA, .)))
te %<>% mutate_all(funs(ifelse(is_na_val(.), NA, .)))

# tr %>% summarise_all(funs(sum(is.na(.))/n()*100)) %>% 
#   gather(key="feature", value="missing_pct") %>% 
#   ggplot(aes(x=reorder(feature,-missing_pct),y=missing_pct)) +
#   geom_bar(stat="identity", fill="steelblue")+
#   labs(y = "missing %", x = "features") +
#   coord_flip() +
#   theme_minimal()

## Simple transformations
#We need to convert some features to their natural representation.

tr %<>% mutate(bounces = replace_na(bounces, 0), 
               newVisits = replace_na(newVisits, 0))

te %<>% mutate(bounces = replace_na(bounces, 0), 
               newVisits = replace_na(newVisits, 0))


tr %<>%
  mutate(date = ymd(date),
         hits = as.integer(hits),
         pageviews = as.integer(pageviews),
         bounces = as.integer(bounces),
         newVisits = as.integer(newVisits),
         transactionRevenue = as.numeric(transactionRevenue))

te %<>%
  mutate(date = ymd(date),
         hits = as.integer(hits),
         pageviews = as.integer(pageviews),
         bounces = as.integer(bounces),
         newVisits = as.integer(newVisits))         

## Target variable
#As a target variable we use **transactionRevenue** which is a sub-column of the **totals** JSON column. It looks like
#this variable is multiplied by $10^6$.

y <- tr$transactionRevenue
tr$transactionRevenue <- NULL
summary(y)

#We can safely replace **NA** values with 0.
y[is.na(y)] <- 0
summary(y)


# #Auto-encoder Code
#
# h2o.no_progress()
# h2o.init(nthreads = 4, max_mem_size = "10G")
# 
# tr_h2o <- as.h2o(tr)
# te_h2o <- as.h2o(te)
# 
# n_ae <- 8
# 
# #Let’s train a simple model, which compresses the input space to `r n_ae` components:
# m_ae <- h2o.deeplearning(training_frame = tr_h2o,
#                          x = 1:ncol(tr_h2o),
#                          autoencoder = T,
#                          activation="Rectifier",
#                          categorical_encoding = "OneHotInternal",
#                          reproducible = TRUE,
#                          seed = 0,
#                          sparse = T,
#                          standardize = TRUE,
#                          hidden = c(32, 16, n_ae, 16, 32),
#                          max_w2 = 5,
#                          epochs = 25)
# tr_ae <- h2o.deepfeatures(m_ae, tr_h2o, layer = 3) %>% as_tibble
# te_ae <- h2o.deepfeatures(m_ae, te_h2o, layer = 3) %>% as_tibble
# 
# #rm(tr_h2o, te_h2o, m_ae); invisible(gc())
# h2o.shutdown(prompt = FALSE)

grp_mean <- function(x, grp) ave(x, grp, FUN = function(x) mean(x, na.rm = TRUE))

idx <- tr$date < ymd("20170701")
id <- te[, "fullVisitorId"]
tri <- 1:nrow(tr)

#tr_temp <- tr %>% bind_cols(tr_ae)
#te_temp <- te %>% bind_cols(te_ae)

tr_te <- tr %>%
  bind_rows(te) %>% 
  mutate(year = year(date) %>% factor(),
         wday = wday(date) %>% factor(),
         hour = hour(as_datetime(visitStartTime)) %>% factor(),
         isMobile = ifelse(isMobile, 1L, 0L),
         isTrueDirect = ifelse(isTrueDirect, 1L, 0L),
         adwordsClickInfo.isVideoAd = ifelse(!adwordsClickInfo.isVideoAd, 0L, 1L)) %>% 
  select(-date, -fullVisitorId, -visitId, -sessionId, -hits, -visitStartTime) %>% 
  mutate_if(is.character, factor) %>% 
  mutate(pageviews_mean_vn = grp_mean(pageviews, visitNumber),
         pageviews_mean_country = grp_mean(pageviews, country),
         pageviews_mean_city = grp_mean(pageviews, city),
         pageviews_mean_dom = grp_mean(pageviews, networkDomain),
         pageviews_mean_ref = grp_mean(pageviews, referralPath)) %T>% 
  glimpse()


#For the **glmnet** model we need a model matrix. We replace **NA** values with zeros, 
#rare factor levels are lumped:
tr_te_ohe <- tr_te %>% 
  mutate_if(is.factor, fct_explicit_na) %>% 
  mutate_if(is.numeric, funs(ifelse(is.na(.), 0L, .))) %>% 
  mutate_if(is.factor, fct_lump, prop = 0.05) %>% 
  select(-adwordsClickInfo.isVideoAd) %>% 
  model.matrix(~.-1, .) %>% 
  scale() %>% 
  round(4)

X <- tr_te_ohe[tri, ]
X_test <- tr_te_ohe[-tri, ]
rm(tr_te_ohe); invisible(gc())

###
###
##################################Block 1 Data Prep######################################
# SVM classifier to classify 0 revenue vs positive revenue
classify <- function(s){ifelse(s==0, 0, 1)}
temp_y <- as.data.frame(sapply(y, classify))
temp_X <- as.data.frame(X)
temp_X <- cbind(temp_X, temp_y)
colnames(temp_X)[73] <- "target"
temp_X$target <- as.factor(temp_X$target)
#temp_X2 <- SMOTE(target ~ ., temp_X, perc.over = 100, perc.under=300)

intrain <- createDataPartition(y = as.integer(temp_X2$target)-1, p= 0.8, list = FALSE)
training <- temp_X2[intrain,]
testing <- temp_X2[-intrain,]
dim(training)
dim(testing)

trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
################################Training Block 1################################

#grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))
svm_Linear_Grid <- train(target ~., data = training, method = "svmRadial", #svmRadial
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    #tuneGrid = grid,
                    tuneLength = 10)

svm_Linear_Grid
#plot(svm_Linear_Grid)
test_pred <- predict(svm_Linear_Grid, newdata = testing)
confusionMatrix(test_pred, testing$target )

fin_predict <- predict(svm_Linear_Grid, newdata = temp_X)
confusionMatrix(fin_predict, temp_X$target )

###
###
##################################Block 2 Data Prep######################################
# SVM classifier to classify 0 revenue vs positive revenue
set.seed(60)
temp_X3 <- SMOTE(target ~ ., temp_X, perc.over = 100, perc.under=200)

intrain2 <- createDataPartition(y = as.integer(temp_X3$target)-1, p= 0.8, list = FALSE)
training2 <- temp_X2[intrain2,]
testing2 <- temp_X2[-intrain2,]
dim(training2)
dim(testing2)

trctrl2 <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
################################Training Block 2################################

grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))
svm_Linear_Grid_tuned <- train(target ~., data = training2, method = "svmRadial", #svmRadial
                         trControl=trctrl2,
                         preProcess = c("center", "scale"),
                         tuneGrid = data.frame(.C = c(.25, .5, 1),
                                               .sigma = .05),
                         tuneLength = 10)

svm_Linear_Grid_tuned
#plot(svm_Linear_Grid)
test_pred2 <- predict(svm_Linear_Grid_tuned, newdata = testing2)
confusionMatrix(test_pred2, testing2$target )

fin_predict2 <- predict(svm_Linear_Grid_tuned, newdata = temp_X)
confusionMatrix(fin_predict2, temp_X$target )

 ###
###
##################################Block 3 Data Prep######################################
set.seed(30)
temp_y2 <- cbind(y, temp_y)
colnames(temp_y2)[2] <- "target"

posX <- temp_X[temp_X$target == 1, 1:72]
posY <- temp_y2[temp_y2$target == 1, 1]
################################Training Block 3################################
## Keras
#For a neural net we can use the same model matrix. Let's create a simple sequential model:
m_nn <- keras_model_sequential() 
m_nn %>% 
  layer_dense(units = 256, activation = "relu", input_shape = ncol(posX)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 256, activation = "sigmoid") %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 1, activation = "linear")

#Next, we compile the model with appropriate parameters:
m_nn %>% compile(loss = "mean_squared_error",
                 metrics = custom_metric("rmse", function(y_true, y_pred) 
                   k_sqrt(metric_mean_squared_error(y_true, y_pred))),
                 optimizer = optimizer_adadelta())

#Then we train the model:
history <- m_nn %>% 
  fit(X, log1p(y), 
      epochs = 100, 
      batch_size = 64, 
      verbose = 0, 
      #validation_split = 0.2,
      callbacks = callback_early_stopping(patience = 5))

#And finally, predictions:
pred_nn_tr <- predict(m_nn, X) %>% c()
pred_nn <- predict(m_nn, X_test) %>% c()

sub <- "keras_gs.csv"
submit(pred_nn)

submit <- . %>% 
  as_tibble() %>% 
  set_names("y") %>% 
  mutate(y = ifelse(y < 0, 0, expm1(y))) %>% 
  bind_cols(id) %>% 
  group_by(fullVisitorId) %>% 
  summarise(y = log1p(sum(y))) %>% 
  right_join(
    read_csv("/Users/p0d00cn/Documents/Learning/Google RP/all/sample_submission.csv"), 
    by = "fullVisitorId") %>% 
  mutate(PredictedLogRevenue = round(y, 5)) %>% 
  select(-y) %>% 
  write_csv(sub)
