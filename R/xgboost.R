rm(list=ls())
library(xgboost)
library(Matrix)
library(caret)
library(car)

box_cox <- function(y,D){
  b=boxcox(y~y, data=D) # ���庯�����ͺ�����
  I=which(b$y==max(b$y))
  return(b$x[I])#lambda=0.83 
}

setwd("C:\\Users\\wuhaoyu\\Desktop\\DG")
# ��������
df_train = read.csv("pfm_train.csv")
df_test = read.csv("pfm_test.csv")
# ���ر�ǩ��ѵ������
labels = df_train['Attrition']
df_train = df_train[-grep('Attrition', colnames(df_train))]
# combine train and test data
df_all = rbind(df_train,df_test)

# ��ϴ����
names(df_all)[1] <- "Age"
df_all <- df_all[, -c(17,22)]

# --
# �����������
df_all <- unite(df_all, "job", Department,JobRole , remove = TRUE)
df_all <- unite(df_all, "marriy_with_travel", BusinessTravel,MaritalStatus, remove = TRUE)
df_all["avg_per_company"] <- df_all["TotalWorkingYears"]/(df_all["NumCompaniesWorked"]+0.5)
df_all["satisfaction"] <- df_all["RelationshipSatisfaction"]*df_all["EnvironmentSatisfaction"]*df_all["JobSatisfaction"]
df_all["manager_relation"] <- df_all["RelationshipSatisfaction"]*df_all["YearsWithCurrManager"]

# # ���ȱ����������
# ohe_feats = c('BusinessTravel',
#               'Department',
#               'EducationField',
#               'Gender',
#               'JobRole',
#               'MaritalStatus',
#               'OverTime'
#               )
# dummies <- dummyVars(~ BusinessTravel+Department+EducationField+
#                       Gender+JobRole+MaritalStatus+OverTime, data = df_all)
chr_var <- c('EducationField',
             'Gender',
             'OverTime',
             'PerformanceRating',
             'marriy_with_travel',
             'job')
ord_var <- c('Education',
             'EnvironmentSatisfaction',
             'JobInvolvement',
             'JobLevel',
             'JobSatisfaction',
             'RelationshipSatisfaction',
             'StockOptionLevel',
             'TrainingTimesLastYear',
             'WorkLifeBalance')
num_var <- c('Age',
             'DistanceFromHome',
             'MonthlyIncome',
             'NumCompaniesWorked',
             'PercentSalaryHike',
             'TotalWorkingYears',
             'YearsAtCompany',
             'YearsInCurrentRole',
             'YearsSinceLastPromotion',
             'YearsWithCurrManager',
             'avg_per_company',
             'satisfaction',
             'manager_relation')
# boxcox�任
df_all[num_var] <- df_all[num_var]+0.5
# df_all[num_var] <- lapply(df_all[num_var], function(x) scale(x, center = T, scale = T))
lambda <- sapply(num_var, function(x) box_cox(df_all[[x,exact=FALSE]],df_all))
lambda_df <- as.data.frame(lambda)
df_all[num_var] <- lapply(num_var, function(x) df_all[[x,exact=FALSE]]^lambda_df[x,"lambda"])

ohe_feats <- c(ord_var, chr_var)
dummies <- dummyVars(~., data = df_all[ohe_feats])
df_all_ohe <- as.data.frame(predict(dummies, newdata = df_all[ohe_feats]))
df_all_combined <- cbind(df_all[,-c(which(colnames(df_all) %in% ohe_feats))],df_all_ohe)
# df_all_combined <- df_all_combined[,c('id',features_selected)] 
# split train and test
X = df_all_combined[df_all_combined$EmployeeNumber %in% df_train$EmployeeNumber,]
y <- recode(labels$Attrition,"'True'=1; 'False'=0")
X_test = df_all_combined[df_all_combined$EmployeeNumber %in% df_test$EmployeeNumber,]
xgb <- xgboost(data = data.matrix(X[,-4]), 
               label = y, 
               eta = 0.3,
               max_depth = 15, 
               nround = 1000, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               eval_metric = "error",
               objective = "binary:logistic",
               # nthread = 3,
               scale_pos_weight = 0.19
)
# �ڲ��Լ�Ԥ���ֵ
y_pred <- predict(xgb, data.matrix(X_test[,-4]))
res <- ifelse(y_pred>0.5,1,0)
(350-sum(res))/350
write.csv(res,'11.csv')
# ģ�͵ľ���
accuracy <- table(res,X_test$Attrition, dnn = c("Ԥ��ֵ","��ʵֵ"));accuracy
accuracy <- mean(res==test$Attrition);accuracy
#����roc��ͼ��
roc <- roc(test$Attrition,pred);roc
plot(roc)

xgb.cv <- xgb.cv(data = data.matrix(X[,-4]), 
               label = y, 
               eta = 0.1,
               max_depth = 2, 
               nround = 500, 
               subsample = 0.8,
               colsample_bytree = 0.2,
               eval_metric = "error",
               objective = "binary:logistic",
               # nthread = 3,
               scale_pos_weight = 1,
               nfold = 5,
               min_child_weight =15,
               gamma =0.7,
               n_estimators=1000,
               lambda =0.1,
               slient = 1,
               alpha = 0.01
)
modelfit(xgb.cv, data.matrix(X[,-4]), y)