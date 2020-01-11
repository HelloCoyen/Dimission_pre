rm(list=ls())
##加载所需包
library(Matrix)
library(foreach)
library(glmnet)
library(caret)
library(cluster)
library(glmnet)
setwd("d:\\Documents\\wuhaoyu\\桌面\\daguan\\data")

# 数据加载，基础预处理
train_Xy <- read.csv("pfm_train.csv", header = TRUE, stringsAsFactors = FALSE)
train_X <- train_Xy[-grep("Attrition", colnames(train_Xy))]
train_y <- train_Xy["Attrition"]
train_y$Attrition <- as.factor(train_y$Attrition)
test_X <- read.csv("pfm_test.csv", header = TRUE, stringsAsFactors = FALSE)
all_X <- rbind(train_X, test_X)

# 数据清洗，列名
names(all_X)
names(all_X)[1] <- "Age"

# 删除无效变量
all_X <- all_X[-grep("Over18|StandardHours", colnames(all_X))]

# 组合变量
all_X <- unite(all_X, "job", Department, JobRole , remove = TRUE)
all_X["avg_per_company"] <- all_X["TotalWorkingYears"]/(all_X["NumCompaniesWorked"]+0.5)
all_X["satisfaction"] <- all_X["RelationshipSatisfaction"]*all_X["EnvironmentSatisfaction"]*all_X["JobSatisfaction"]
all_X["manager_relation"] <- all_X["RelationshipSatisfaction"]*all_X["YearsWithCurrManager"]

# 特征类型转换
ord_var <- c("BusinessTravel", "Education", "EnvironmentSatisfaction", "JobInvolvement",
             "JobLevel", "JobSatisfaction", "PerformanceRating", "RelationshipSatisfaction",
             "TrainingTimesLastYear", "WorkLifeBalance",'job')
all_X[ord_var] <- lapply(all_X[ord_var], function(x) as.factor(x))

# 全变量
all_columns = names(all_X)

# 数值变量
num_var <- c("Age", "DistanceFromHome", "MonthlyIncome", "NumCompaniesWorked",
             "PercentSalaryHike", "StockOptionLevel", "TotalWorkingYears", "YearsAtCompany",
             "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
             "avg_per_company", "satisfaction", "manager_relation")

# # ohe_feats = c('BusinessTravel',
# #               'EducationField',
# #               'Gender',
# #               'MaritalStatus',
# #               'OverTime',
# #               'job'
# # )
# # all_X[ohe_feats] <- lapply(all_X[ohe_feats], function(x) as.factor(x))
# 
# num_X <- all_X[-which(names(all_X)%in%ohe_feats)]
# num_X <- num_X[-grep("EmployeeNumber", colnames(num_X))] 
# num_X <- lapply(num_X, function(x) scale(x, center = T, scale = T))
# num_X <-  as.data.frame(num_X)
# df_all_combined <- cbind(all_X["EmployeeNumber"],num_X)
# # 非数值变量
unnum_var <- all_columns[-which(all_columns%in%num_var)]

# 特征类型转换
dummies <- dummyVars(~BusinessTravel+job+Education+EducationField+
                       EnvironmentSatisfaction+Gender+JobInvolvement+
                       JobLevel+JobSatisfaction+MaritalStatus+OverTime+
                       PerformanceRating+RelationshipSatisfaction+TrainingTimesLastYear+
                       WorkLifeBalance, data = all_X)
df_all_ohe <- as.data.frame(predict(dummies, newdata = all_X))
df_all_combined <- cbind(all_X[,c(which(colnames(all_X) %in% num_var))], df_all_ohe)
df_all_combined <- lapply(df_all_combined, function(x) scale(x, center = T, scale = T))
df_all_combined <- cbind(all_X["EmployeeNumber"],df_all_combined)

tr_x <- df_all_combined[which(df_all_combined$EmployeeNumber%in%train_X$EmployeeNumber),]
tr_x <- tr_x[-grep("EmployeeNumber", colnames(tr_x))]
tr_x <- as.matrix(tr_x)
tr_y <- as.matrix(train_y)
t_x <- df_all_combined[which(df_all_combined$EmployeeNumber%in%test_X$EmployeeNumber),]
t_x <- t_x[-grep("EmployeeNumber", colnames(t_x))]
t_x <- as.matrix(t_x)

cvfit=cv.glmnet(tr_x,tr_y,family="binomial",alpha=1,type.measure="mse")     #10折交叉验证
plot(cvfit)      #Lambda与变量数目对应走势

coef(cvfit,s="lambda.1se")   #变量系数

b=glmnet(tr_x,tr_y,alpha=1,family="binomial",lambda=cvfit$lambda)

plot(b,xvar="lambda",label=TRUE)    #Lasso系数解路径

out=glmnet(tr_x,tr_y,alpha=1,family="binomial")

lasso.pred00=predict(cvfit,newx=tr_x,type="class",s="lambda.min")    #训练集的预测结果
lasso.pred00 <- as.numeric(lasso.pred00)
sum(lasso.pred00)
lasso.pred02=predict(cvfit,newx=t_x,type="class",s="lambda.1se")      #测试集的预测结果
lasso.pred02 <- as.numeric(lasso.pred02)
sum(lasso.pred02)

ST0=y    #训练数据集的真实结果

ST1=c[,1]      #测试数据集的真实结果

table(ST0,lasso.pred00)

table(ST1,lasso.pred02)