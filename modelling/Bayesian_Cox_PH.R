
rm(list=ls())
library(survival)
library(spBayesSurv)
library(coda)
library(dplyr)
library(tidyr)
library(DBI)
library(pool)


## Data Import
loan_data <- read.csv('./data/loans_labelled_train.csv')
#OBSERVATION_LOAN_AGE
loan_data <- loan_data[loan_data$REMAINING_SURVIVAL_TIME > 0, ]


# only use subsample ...
train_1k_ids <- read.csv('./data/train_ids_1k.csv')
train_10k_ids <- read.csv('./data/train_ids_10k.csv')
loan_data <- loan_data[loan_data$LOAN_SEQ_NUMBER %in% train_1k_ids$LOAN_SEQ_NUMBER, ]


surv_response <- Surv(loan_data$REMAINING_SURVIVAL_TIME, loan_data$DEFAULT)

loan_test_data <- read.csv('./data/loans_labelled_test.csv')

# features as in Blumenstock 2020
features_list <- c('CREDIT_SCORE','ORIG_DTI_RATIO','ORIG_UPB','ORIG_LTV','ORIG_INTEREST_RATE','UPB_Share',"TIMES_NOT_BEING_DELINQUENT","TIMES_BEING_DELINQUENT_30_DAYS","TIMES_BEING_DELINQUENT_60_DAYS")
 

scaled_object <- scale(loan_data[, features_list])
X_train <- as.data.frame(scaled_object)

X_test = as.data.frame(scale(loan_test_data[, features_list], center=attr(scaled_object, "scaled:center"), 
                      scale=attr(scaled_object, "scaled:scale")))


# converts the table to wide format with the ID as index
convert_preds_to_df <- function(predictions, ids, periods, pred_name='PRED_MEAN'){
  df <- data.frame(predictions)
  colnames(df) <- ids
  df['PERIOD'] <- periods
  
  df <- df %>% gather(key='ID', value = 'PRED', -'PERIOD')
  
  # rearrange columns
  df <- df[c('ID','PERIOD', 'PRED')]
  
  colnames(df) <- c('ID', 'PERIOD', pred_name)
  
  return(df)
}

###############################################################
# Independent Cox PH
###############################################################
# MCMC parameters
nburn=500; nsave=3000; nskip=0;
# Note larger nburn, nsave and nskip should be used in practice.
mcmc=list(nburn=nburn, nsave=nsave, nskip=nskip, ndisplay=500);
prior = list(M=10, r0=1);
# Fit the Cox PH model
print("Fitting Model")
res1 = indeptCoxph(formula = surv_response~CREDIT_SCORE+ORIG_DTI_RATIO+ORIG_UPB+ORIG_LTV+ORIG_INTEREST_RATE+UPB_Share+TIMES_NOT_BEING_DELINQUENT+TIMES_BEING_DELINQUENT_30_DAYS+TIMES_BEING_DELINQUENT_60_DAYS, data=X_train, 
                   prior=prior, mcmc=mcmc);

sfit1=summary(res1); sfit1;

#write.csv(sfit1$coeff, 'Cox_Coef.csv')

#saveRDS(res1, file = 'bayes_cox_ph.RDS')
#res1 <- readRDS('bayes_cox_ph.RDS')


#X_test <- loan_test_data[, c('CREDIT_SCORE', 'ORIG_COMBINED_LTV','ORIG_DTI_RATIO', 'ORIG_UPB', 'ORIG_LTV', 'ORIG_INTEREST_RATE', 'NUMBER_BORROWERS')]


pool_predictions <- dbPool(RSQLite::SQLite(),
                           dbname = './predictions/test_evaluation_1k_cox.sqlite')


prediction_periods <- seq(0,47)

idx_start <- 1
idx_end <- 1
# use parts of 1000 samples to predict
sample_size <- 1000

while(idx_start < nrow(X_test)){
  print(paste("Iteration", idx_start))
  
  idx_end <- idx_start + sample_size
  
  # in case in last
  if(idx_end > nrow(X_test)){
    idx_end <- nrow(X_test)
  }
  
  sample_IDs <- seq(idx_start, idx_end)
  
  preds_95 <- GetCurves(x = res1, xnewdata =X_test[sample_IDs ,], CI = 0.95, PLOT = FALSE, tgrid = prediction_periods)
  preds_80 <- GetCurves(x = res1, xnewdata =X_test[sample_IDs ,], CI = 0.8, PLOT = FALSE, tgrid = prediction_periods)
  
  preds_80_lower <- convert_preds_to_df(preds_80$Shatlow, sample_IDs, prediction_periods, pred_name = 'PRED_80_LOW')
  preds_80_upper <- convert_preds_to_df(preds_80$Shatup, sample_IDs, prediction_periods, pred_name = 'PRED_80_HIGH')
  
  preds_95_lower <- convert_preds_to_df(preds_95$Shatlow, sample_IDs, prediction_periods, pred_name = 'PRED_95_LOW')
  preds_95_upper <- convert_preds_to_df(preds_95$Shatup, sample_IDs, prediction_periods, pred_name = 'PRED_95_HIGH')
  
  preds_mean <- convert_preds_to_df(preds_95$Shat, sample_IDs, prediction_periods, pred_name = 'PRED_MEAN')
  
  final_S_preds <- merge(preds_80_lower, preds_80_upper, by=c('ID', 'PERIOD'))
  final_S_preds <- merge(final_S_preds, preds_95_lower, by=c('ID', 'PERIOD'))
  final_S_preds <- merge(final_S_preds, preds_95_upper, by=c('ID', 'PERIOD'))
  final_S_preds <- merge(final_S_preds, preds_mean, by=c('ID', 'PERIOD'))
  
  # sort values
  final_S_preds <- final_S_preds %>% arrange(ID, PERIOD)
  
  final_S_preds['MODEL'] <- 'COX_BAYES'
  
  # rearrange columns
  final_S_preds <- final_S_preds[c("ID","PERIOD","MODEL","PRED_80_LOW","PRED_80_HIGH","PRED_95_LOW","PRED_95_HIGH","PRED_MEAN")]

  # write predictions to SQL
  dbWriteTable(pool_predictions, "TB02_PREDICTIONS", final_S_preds, append=TRUE,overwrite = FALSE)
  
  idx_start <- idx_start + sample_size
}

poolClose(pool_predictions)


